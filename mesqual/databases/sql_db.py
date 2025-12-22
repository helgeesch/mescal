import hashlib
import json
import os
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse

import pandas as pd

try:
    import sqlalchemy as sa
    from sqlalchemy import create_engine, text, MetaData, inspect
    from sqlalchemy.engine import Engine
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False

try:
    import geopandas as gpd
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False

from mesqual.typevars import DatasetType, FlagType, DatasetConfigType
from mesqual.databases.database import Database


class SQLDatabase(Database):
    """SQL-based database implementation with automatic table management.

    Features:
    - Automatic table creation based on DataFrame schemas
    - Schema sharing for DataFrames with identical structures
    - Support for GeoDataFrames with multiple geometry columns
    - Automatic schema evolution (adding new columns)
    - Efficient deletion of specific dataset/flag combinations
    """

    def __init__(self, connection_string: str, table_prefix: str = "mesqual"):
        """Initialize SQL database.

        Args:
            connection_string: SQLAlchemy connection string (e.g., "sqlite:///cache.db")
            table_prefix: Prefix for auto-generated table names

        Raises:
            ImportError: If sqlalchemy is not available
        """
        if not SQLALCHEMY_AVAILABLE:
            raise ImportError("sqlalchemy is required for SQL database support. Install with: pip install sqlalchemy")
        self.connection_string = connection_string
        self.table_prefix = table_prefix
        self.engine = create_engine(connection_string)
        self._metadata = MetaData()
        self._inspector = inspect(self.engine)

        # Ensure database directory exists for file-based databases
        if connection_string.startswith('sqlite:///'):
            db_path = connection_string.replace('sqlite:///', '')
            os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)

    def get(
            self,
            dataset: DatasetType,
            flag: FlagType,
            config: DatasetConfigType,
            **kwargs
    ) -> pd.Series | pd.DataFrame:
        """Retrieve data from SQL database."""
        dataset_key = self._get_dataset_key(dataset, flag, config, **kwargs)
        table_name = self._find_table_for_key(dataset_key)

        if not table_name:
            raise KeyError(f"No data found for key: {dataset_key}")

        # Check if this is a GeoDataFrame table
        if self._is_geo_table(table_name):
            if not GEOPANDAS_AVAILABLE:
                raise ImportError("geopandas is required for geometry data")
            return gpd.read_postgis(
                f"SELECT * FROM {table_name} WHERE dataset_key = '{dataset_key}'",
                self.engine
            )
        else:
            return pd.read_sql(
                f"SELECT * FROM {table_name} WHERE dataset_key = '{dataset_key}'",
                self.engine,
                index_col=self._get_index_columns(table_name)
            )

    def set(
            self,
            dataset: DatasetType,
            flag: FlagType,
            config: DatasetConfigType,
            value: pd.DataFrame | pd.Series,
            **kwargs
    ):
        """Store data in SQL database with automatic table management."""
        if isinstance(value, pd.Series):
            value = value.to_frame()

        dataset_key = self._get_dataset_key(dataset, flag, config, **kwargs)

        # Delete existing data for this key
        self._delete_by_key(dataset_key)

        # Get or create table for this schema
        table_name = self._get_or_create_table(value, dataset_key)

        # Prepare DataFrame for storage
        value_to_store = value.copy()
        value_to_store['dataset_key'] = dataset_key

        # Store data
        if hasattr(value, 'geometry') and GEOPANDAS_AVAILABLE:
            # GeoDataFrame - use to_postgis
            value_to_store.to_postgis(
                table_name,
                self.engine,
                if_exists='append',
                index=True
            )
        else:
            # Regular DataFrame
            value_to_store.to_sql(
                table_name,
                self.engine,
                if_exists='append',
                index=True
            )

    def key_is_up_to_date(
            self,
            dataset: DatasetType,
            flag: FlagType,
            config: DatasetConfigType,
            **kwargs
    ) -> bool:
        """Check if data exists for the given key."""
        dataset_key = self._get_dataset_key(dataset, flag, config, **kwargs)
        return self._key_exists(dataset_key)

    def delete(
            self,
            dataset: Optional[DatasetType] = None,
            flag: Optional[FlagType] = None,
            config: Optional[DatasetConfigType] = None,
            **kwargs
    ):
        """Delete data matching the given criteria."""
        conditions = []

        if dataset is not None:
            conditions.append(f"dataset_key LIKE '{dataset.name}_%'")

        if flag is not None:
            conditions.append(f"dataset_key LIKE '%_{str(flag)}_%'")

        if config is not None:
            config_hash = self._get_config_hash(config)
            if config_hash:
                conditions.append(f"dataset_key LIKE '%_config_{config_hash}_%'")

        if kwargs:
            kwargs_hash = self._get_kwargs_hash(kwargs)
            conditions.append(f"dataset_key LIKE '%_kwargs_{kwargs_hash}'")

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        # Delete from all tables
        for table_name in self._get_all_table_names():
            with self.engine.connect() as conn:
                conn.execute(text(f"DELETE FROM {table_name} WHERE {where_clause}"))
                conn.commit()

    def list_keys(
            self,
            dataset: Optional[DatasetType] = None,
            flag: Optional[FlagType] = None
    ) -> List[str]:
        """List all dataset keys matching the given criteria."""
        keys = set()

        for table_name in self._get_all_table_names():
            query = f"SELECT DISTINCT dataset_key FROM {table_name}"
            conditions = []

            if dataset is not None:
                conditions.append(f"dataset_key LIKE '{dataset.name}_%'")

            if flag is not None:
                conditions.append(f"dataset_key LIKE '%_{str(flag)}_%'")

            if conditions:
                query += " WHERE " + " AND ".join(conditions)

            with self.engine.connect() as conn:
                result = conn.execute(text(query))
                keys.update(row[0] for row in result)

        return sorted(list(keys))

    # Helper methods

    def _get_dataset_key(self, dataset: DatasetType, flag: FlagType, config: DatasetConfigType, **kwargs) -> str:
        """Generate unique key for dataset/flag/config/kwargs combination."""
        components = [dataset.name, str(flag)]

        config_hash = self._get_config_hash(config)
        if config_hash:
            components.append(f"config_{config_hash}")

        kwargs_hash = self._get_kwargs_hash(kwargs)
        if kwargs_hash:
            components.append(f"kwargs_{kwargs_hash}")

        return "_".join(components)

    def _get_config_hash(self, config: DatasetConfigType = None) -> str:
        """Generate hash for config object."""
        if config is None:
            return ""

        attrs = {
            name: getattr(config, name)
            for name in dir(config)
            if not name.startswith('_') and not callable(getattr(config, name))
        }

        sorted_items = sorted(attrs.items())
        config_str = str(sorted_items)
        return str(abs(hash(config_str)))

    def _get_kwargs_hash(self, kwargs: dict) -> str:
        """Generate hash for kwargs dict."""
        if not kwargs:
            return ""

        str_dict = {str(k): str(v) for k, v in kwargs.items()}
        sorted_items = sorted(str_dict.items())
        return str(abs(hash(str(sorted_items))))

    def _get_schema_hash(self, df: pd.DataFrame) -> str:
        """Generate hash for DataFrame schema."""
        schema_info = {
            'columns': list(df.columns),
            'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'index_names': df.index.names if hasattr(df.index, 'names') else [df.index.name],
            'has_geometry': hasattr(df, 'geometry')
        }

        schema_str = json.dumps(schema_info, sort_keys=True)
        return hashlib.md5(schema_str.encode()).hexdigest()[:12]

    def _get_or_create_table(self, df: pd.DataFrame, dataset_key: str) -> str:
        """Get existing table or create new one for the DataFrame schema."""
        schema_hash = self._get_schema_hash(df)
        table_name = f"{self.table_prefix}_{schema_hash}"

        if not self._table_exists(table_name):
            self._create_table_from_dataframe(table_name, df)
        else:
            self._ensure_schema_compatibility(table_name, df)

        return table_name

    def _create_table_from_dataframe(self, table_name: str, df: pd.DataFrame):
        """Create table with schema matching the DataFrame."""
        # Let pandas/sqlalchemy handle the table creation
        sample_df = df.head(0).copy()  # Empty DataFrame with same schema
        sample_df['dataset_key'] = ''  # Add dataset_key column

        if hasattr(df, 'geometry') and GEOPANDAS_AVAILABLE:
            sample_df.to_postgis(table_name, self.engine, if_exists='replace', index=True)
        else:
            sample_df.to_sql(table_name, self.engine, if_exists='replace', index=True)

    def _ensure_schema_compatibility(self, table_name: str, df: pd.DataFrame):
        """Ensure table schema can accommodate the DataFrame."""
        existing_columns = set(self._get_table_columns(table_name))
        df_columns = set(df.columns) | {'dataset_key'}

        missing_columns = df_columns - existing_columns
        if missing_columns:
            # Add missing columns (simplified - in practice you'd want proper ALTER TABLE)
            for col in missing_columns:
                if col != 'dataset_key':
                    dtype = 'TEXT'  # Simplified - could infer proper type
                    with self.engine.connect() as conn:
                        conn.execute(text(f"ALTER TABLE {table_name} ADD COLUMN {col} {dtype}"))
                        conn.commit()

    def _table_exists(self, table_name: str) -> bool:
        """Check if table exists in database."""
        return table_name in self._inspector.get_table_names()

    def _get_table_columns(self, table_name: str) -> List[str]:
        """Get column names for table."""
        return [col['name'] for col in self._inspector.get_columns(table_name)]

    def _get_index_columns(self, table_name: str) -> Optional[List[str]]:
        """Get index column names for table."""
        # Simplified - would need more logic for MultiIndex
        return None

    def _is_geo_table(self, table_name: str) -> bool:
        """Check if table contains geometry columns."""
        if not GEOPANDAS_AVAILABLE:
            return False

        columns = self._get_table_columns(table_name)
        return any('geometry' in col.lower() for col in columns)

    def _find_table_for_key(self, dataset_key: str) -> Optional[str]:
        """Find which table contains the given dataset key."""
        for table_name in self._get_all_table_names():
            if self._key_exists_in_table(dataset_key, table_name):
                return table_name
        return None

    def _key_exists(self, dataset_key: str) -> bool:
        """Check if dataset key exists in any table."""
        return self._find_table_for_key(dataset_key) is not None

    def _key_exists_in_table(self, dataset_key: str, table_name: str) -> bool:
        """Check if dataset key exists in specific table."""
        with self.engine.connect() as conn:
            result = conn.execute(
                text(f"SELECT 1 FROM {table_name} WHERE dataset_key = :key LIMIT 1"),
                {'key': dataset_key}
            )
            return result.fetchone() is not None

    def _delete_by_key(self, dataset_key: str):
        """Delete all data for a specific dataset key."""
        for table_name in self._get_all_table_names():
            with self.engine.connect() as conn:
                conn.execute(
                    text(f"DELETE FROM {table_name} WHERE dataset_key = :key"),
                    {'key': dataset_key}
                )
                conn.commit()

    def _get_all_table_names(self) -> List[str]:
        """Get all table names with our prefix."""
        all_tables = self._inspector.get_table_names()
        return [t for t in all_tables if t.startswith(self.table_prefix)]