import hashlib
import json
from typing import Optional, List, Dict, Any, Union
from urllib.parse import urlparse

from datetime import datetime
import numpy as np
import pandas as pd
from shapely.geometry import shape
import geopandas as gpd

try:
    from bson import ObjectId
    import pymongo
    from pymongo import MongoClient, ASCENDING, DESCENDING
    from pymongo.errors import ServerSelectionTimeoutError, DuplicateKeyError
    PYMONGO_AVAILABLE = True
except ImportError:
    PYMONGO_AVAILABLE = False

from mesqual.typevars import DatasetType, FlagType, DatasetConfigType
from mesqual.databases.database import Database


class MongoDatabase(Database):
    """MongoDB-based database implementation with automatic collection management.

    This implementation stores DataFrames as MongoDB documents with automatic
    collection creation, indexing, and support for geospatial data through GeoJSON.
    Each dataset/flag/config combination is stored as separate documents within
    collections organized by schema similarity.

    Features:
        - Automatic collection creation based on DataFrame schemas
        - Compound indexes for efficient querying
        - GeoJSON support for spatial data with 2dsphere indexes
        - JSON serialization for complex pandas data types
        - Connection pooling and error handling
        - Support for MongoDB replica sets and authentication

    Attributes:
        connection_string (str): MongoDB connection URI
        collection_prefix (str): Prefix for auto-generated collection names
        client (MongoClient): MongoDB client connection
        database: MongoDB database instance

    Example:
        Basic usage:

        >>> db = MongoDatabase("mongodb://localhost:27017/cache_db")
        >>> # Collections will be auto-created as: energy_models_schema_hash
        >>> db.set(dataset, flag, config, my_dataframe)
        >>> data = db.get(dataset, flag, config)

        With connection options:

        >>> db = MongoDatabase(
        ...     "mongodb://user:pass@replica1:27017,replica2:27017/cache_db?replicaSet=rs0",
        ...     collection_prefix="mesqual_cache",
        ...     connect_timeout=5000
        ... )
    """

    def __init__(
        self,
        connection_string: str,
        collection_prefix: str = "mesqual",
        connect_timeout: int = 10000,
        server_selection_timeout: int = 5000
    ):
        """Initialize MongoDB database connection.

        Args:
            connection_string: MongoDB connection URI (e.g., "mongodb://localhost:27017/mydb")
            collection_prefix: Prefix for auto-generated collection names
            connect_timeout: Connection timeout in milliseconds
            server_selection_timeout: Server selection timeout in milliseconds

        Raises:
            ImportError: If pymongo is not available
            ServerSelectionTimeoutError: If cannot connect to MongoDB server
        """
        if not PYMONGO_AVAILABLE:
            raise ImportError(
                "pymongo and bson are required for MongoDB support. Install with: pip install pymongo bson"
            )

        self.connection_string = connection_string
        self.collection_prefix = collection_prefix

        # Parse connection string to extract database name
        parsed = urlparse(connection_string)
        self.database_name = parsed.path.lstrip('/') or 'mesqual_default'

        # Initialize MongoDB client with timeout settings
        self.client = MongoClient(
            connection_string,
            connectTimeoutMS=connect_timeout,
            serverSelectionTimeoutMS=server_selection_timeout
        )

        # Test connection
        try:
            self.client.server_info()
        except ServerSelectionTimeoutError as e:
            raise ServerSelectionTimeoutError(f"Cannot connect to MongoDB at {connection_string}: {e}")

        self.database = self.client[self.database_name]

    def get(
        self,
        dataset: DatasetType,
        flag: FlagType,
        config: DatasetConfigType,
        **kwargs
    ) -> pd.Series | pd.DataFrame:
        """Retrieve data from MongoDB collection.

        Args:
            dataset: Dataset type identifier
            flag: Processing flag or stage identifier
            config: Configuration object containing processing parameters
            **kwargs: Additional keyword arguments for cache key generation

        Returns:
            pd.Series | pd.DataFrame: The cached data loaded from MongoDB

        Raises:
            KeyError: If no data found for the given key combination
            ValueError: If data format is invalid or corrupted

        Example:
            >>> data = db.get(NetworkDataset.TOPOLOGY, "processed", config)
            >>> isinstance(data, pd.DataFrame)
            True
        """
        dataset_key = self._get_dataset_key(dataset, flag, config, **kwargs)
        collection = self._find_collection_for_key(dataset_key)

        if not collection:
            raise KeyError(f"No data found for key: {dataset_key}")

        # Query for documents with this dataset key
        documents = list(collection.find({"dataset_key": dataset_key}))
        if not documents:
            raise KeyError(f"No data found for key: {dataset_key}")

        return self._documents_to_dataframe(documents)

    def set(
        self,
        dataset: DatasetType,
        flag: FlagType,
        config: DatasetConfigType,
        value: Union[pd.DataFrame, pd.Series],
        **kwargs
    ):
        """Store data in MongoDB collection with automatic collection management.

        Args:
            dataset: Dataset type identifier
            flag: Processing flag or stage identifier
            config: Configuration object containing processing parameters
            value: Data to store (pandas Series or DataFrame)
            **kwargs: Additional keyword arguments for cache key generation

        Raises:
            ValueError: If data cannot be serialized to MongoDB format

        Example:
            >>> df = pd.DataFrame({'value': [1, 2, 3]})
            >>> db.set(NetworkDataset.BUSES, "raw", config, df)
        """
        if isinstance(value, pd.Series):
            value = value.to_frame()

        dataset_key = self._get_dataset_key(dataset, flag, config, **kwargs)

        # Delete existing data for this key
        self._delete_by_key(dataset_key)

        # Get or create collection for this schema
        collection = self._get_or_create_collection(value, dataset_key)

        # Convert DataFrame to MongoDB documents
        documents = self._dataframe_to_documents(value, dataset_key)

        # Insert documents
        if documents:
            collection.insert_many(documents)

    def key_is_up_to_date(
        self,
        dataset: DatasetType,
        flag: FlagType,
        config: DatasetConfigType,
        **kwargs
    ) -> bool:
        """Check if data exists for the given key.

        Args:
            dataset: Dataset type identifier
            flag: Processing flag or stage identifier
            config: Configuration object containing processing parameters
            **kwargs: Additional keyword arguments for cache key generation

        Returns:
            bool: True if data exists for the given key combination

        Example:
            >>> exists = db.key_is_up_to_date(dataset, flag, config)
            >>> print(f"Data exists: {exists}")
        """
        dataset_key = self._get_dataset_key(dataset, flag, config, **kwargs)
        return self._key_exists(dataset_key)

    def delete(
        self,
        dataset: Optional[DatasetType] = None,
        flag: Optional[FlagType] = None,
        config: Optional[DatasetConfigType] = None,
        **kwargs
    ):
        """Delete data matching the given criteria from all collections.

        Uses MongoDB query patterns to match and delete documents. If no parameters
        are provided, this will delete all data in collections with the configured prefix.

        Args:
            dataset: Dataset type to filter by (None matches all datasets)
            flag: Flag to filter by (None matches all flags)
            config: Configuration to filter by (None matches all configs)
            **kwargs: Additional keyword arguments to filter by

        Warning:
            Be careful when calling without parameters as it will delete all data.

        Example:
            >>> # Delete all data for a specific dataset
            >>> db.delete(dataset=NetworkDataset.BUSES)
            >>> # Delete specific dataset/flag combination
            >>> db.delete(dataset=NetworkDataset.BUSES, flag="processed")
        """
        # Build query filter based on provided parameters
        query_filter = self._build_delete_filter(dataset, flag, config, **kwargs)

        # Delete from all relevant collections
        for collection_name in self._get_all_collection_names():
            collection = self.database[collection_name]
            result = collection.delete_many(query_filter)
            if result.deleted_count > 0:
                print(f"Deleted {result.deleted_count} documents from {collection_name}")

    def list_keys(
        self,
        dataset: Optional[DatasetType] = None,
        flag: Optional[FlagType] = None
    ) -> List[str]:
        """List all dataset keys matching the given criteria.

        Args:
            dataset: Dataset type to filter by (None matches all datasets)
            flag: Flag to filter by (None matches all flags)

        Returns:
            List[str]: Sorted list of unique dataset keys matching the criteria

        Example:
            >>> keys = db.list_keys(dataset=NetworkDataset.BUSES)
            >>> print(f"Found {len(keys)} cached datasets")
            >>> for key in keys:
            ...     print(f"  - {key}")
        """
        keys = set()

        # Build query filter
        query_filter = {}
        if dataset is not None:
            query_filter["dataset_key"] = {"$regex": f"^{dataset.name}_"}
        if flag is not None:
            if "dataset_key" in query_filter:
                # Combine with existing regex
                query_filter["dataset_key"]["$regex"] = f"^{dataset.name}_{flag}_"
            else:
                query_filter["dataset_key"] = {"$regex": f"_{flag}_"}

        # Query all collections
        for collection_name in self._get_all_collection_names():
            collection = self.database[collection_name]
            distinct_keys = collection.distinct("dataset_key", query_filter)
            keys.update(distinct_keys)

        return sorted(list(keys))

    def close(self):
        """Close MongoDB connection.

        Good practice to call this when done with the database to free resources.

        Example:
            >>> db = MongoDatabase("mongodb://localhost:27017/cache")
            >>> try:
            ...     # Use database
            ...     pass
            ... finally:
            ...     db.close()
        """
        if hasattr(self, 'client'):
            self.client.close()

    # Helper methods

    def _get_dataset_key(self, dataset: DatasetType, flag: FlagType, config: DatasetConfigType, **kwargs) -> str:
        """Generate unique key for dataset/flag/config/kwargs combination.

        Args:
            dataset: Dataset type identifier
            flag: Processing flag identifier
            config: Configuration object (optional)
            **kwargs: Additional keyword arguments

        Returns:
            str: Unique string identifier for this data combination
        """
        components = [dataset.name, str(flag)]

        config_hash = self._get_config_hash(config)
        if config_hash:
            components.append(f"config_{config_hash}")

        kwargs_hash = self._get_kwargs_hash(kwargs)
        if kwargs_hash:
            components.append(f"kwargs_{kwargs_hash}")

        return "_".join(components)

    def _get_config_hash(self, config: DatasetConfigType = None) -> str:
        """Generate hash for config object.

        Args:
            config: Configuration object to hash

        Returns:
            str: Hash string representing the configuration, empty if None
        """
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
        """Generate hash for kwargs dict.

        Args:
            kwargs: Dictionary of keyword arguments to hash

        Returns:
            str: Hash string representing the kwargs, empty if empty dict
        """
        if not kwargs:
            return ""

        str_dict = {str(k): str(v) for k, v in kwargs.items()}
        sorted_items = sorted(str_dict.items())
        return str(abs(hash(str(sorted_items))))

    def _get_schema_hash(self, df: pd.DataFrame) -> str:
        """Generate hash for DataFrame schema.

        Args:
            df: DataFrame to generate schema hash for

        Returns:
            str: MD5 hash representing the DataFrame schema
        """
        schema_info = {
            'columns': list(df.columns),
            'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'index_names': df.index.names if hasattr(df.index, 'names') else [df.index.name],
            'has_geometry': hasattr(df, 'geometry')
        }

        schema_str = json.dumps(schema_info, sort_keys=True)
        return hashlib.md5(schema_str.encode()).hexdigest()[:12]

    def _get_or_create_collection(self, df: pd.DataFrame, dataset_key: str) -> 'pymongo.collection.Collection':
        """Get existing collection or create new one for the DataFrame schema.

        Args:
            df: DataFrame to store
            dataset_key: Dataset key for this data

        Returns:
            pymongo.collection.Collection: MongoDB collection instance
        """
        schema_hash = self._get_schema_hash(df)
        collection_name = f"{self.collection_prefix}_{schema_hash}"
        collection = self.database[collection_name]

        # Create indexes if collection is new
        if collection_name not in self.database.list_collection_names():
            self._create_collection_indexes(collection, df)

        return collection

    def _create_collection_indexes(self, collection: 'pymongo.collection.Collection', df: pd.DataFrame):
        """Create appropriate indexes for the collection.

        Args:
            collection: MongoDB collection to create indexes on
            df: DataFrame to analyze for index creation
        """
        # Always create index on dataset_key for efficient queries
        collection.create_index([("dataset_key", ASCENDING)])

        # Create geospatial index if DataFrame has geometry
        if hasattr(df, 'geometry'):
            # Create 2dsphere index for geospatial queries
            geometry_fields = [col for col in df.columns if 'geometry' in col.lower()]
            for geo_field in geometry_fields:
                collection.create_index([(geo_field, "2dsphere")])

        # Create compound index for efficient dataset_key + time-based queries
        if isinstance(df.index, pd.DatetimeIndex):
            collection.create_index([("dataset_key", ASCENDING), ("_index", ASCENDING)])

    def _dataframe_to_documents(self, df: pd.DataFrame, dataset_key: str) -> List[Dict[str, Any]]:
        """Convert DataFrame to list of MongoDB documents.

        Args:
            df: DataFrame to convert
            dataset_key: Dataset key to include in each document

        Returns:
            List[Dict[str, Any]]: List of documents ready for MongoDB insertion
        """
        documents = []

        for idx, row in df.iterrows():
            doc = {"dataset_key": dataset_key}

            # Handle index (can be single value or tuple for MultiIndex)
            if isinstance(idx, tuple):
                for i, index_name in enumerate(df.index.names):
                    idx_name = index_name or f"level_{i}"
                    doc[f"_index_{idx_name}"] = self._serialize_value(idx[i])
            else:
                index_name = df.index.name or "_index"
                doc[index_name] = self._serialize_value(idx)

            # Handle columns
            for col, value in row.items():
                if hasattr(df, 'geometry') and col == 'geometry':
                    # Convert geometry to GeoJSON
                    if pd.notna(value):
                        doc[col] = json.loads(gpd.GeoSeries([value]).to_json())['features'][0]['geometry']
                else:
                    doc[col] = self._serialize_value(value)

            documents.append(doc)

        return documents

    def _documents_to_dataframe(self, documents: List[Dict[str, Any]]) -> pd.DataFrame:
        """Convert MongoDB documents back to DataFrame.

        Args:
            documents: List of MongoDB documents

        Returns:
            pd.DataFrame: Reconstructed DataFrame

        Raises:
            ValueError: If documents cannot be converted to DataFrame
        """
        if not documents:
            raise ValueError("Cannot create DataFrame from empty document list")

        # Separate index columns from data columns
        first_doc = documents[0]
        index_cols = [k for k in first_doc.keys() if k.startswith('_index') or k in ['_id', 'dataset_key']]
        data_cols = [k for k in first_doc.keys() if k not in index_cols]

        # Build DataFrame data
        df_data = {}
        for col in data_cols:
            df_data[col] = [self._deserialize_value(doc.get(col)) for doc in documents]

        # Handle index reconstruction
        index_data = []
        for doc in documents:
            # Find index columns (excluding _id and dataset_key)
            idx_cols = [k for k in doc.keys() if k.startswith('_index')]
            if len(idx_cols) == 1:
                # Single index
                index_data.append(self._deserialize_value(doc[idx_cols[0]]))
            elif len(idx_cols) > 1:
                # MultiIndex
                index_data.append(tuple(self._deserialize_value(doc[col]) for col in sorted(idx_cols)))
            else:
                # Use document position as index
                index_data.append(len(index_data))

        df = pd.DataFrame(df_data, index=index_data)

        # Convert to GeoDataFrame if geometry columns exist
        geometry_cols = [col for col in df.columns if 'geometry' in col.lower()]
        if geometry_cols:
            for geo_col in geometry_cols:
                df[geo_col] = df[geo_col].apply(
                    lambda x: shape(x) if x and isinstance(x, dict) else x
                )
            df = gpd.GeoDataFrame(df, geometry=geometry_cols[0])

        return df

    def _serialize_value(self, value: Any) -> Any:
        """Serialize pandas value for MongoDB storage.

        Args:
            value: Value to serialize

        Returns:
            Any: MongoDB-compatible value
        """
        if pd.isna(value):
            return None
        elif isinstance(value, (pd.Timestamp, datetime)):
            return value
        elif isinstance(value, np.datetime64):
            return pd.Timestamp(value)
        elif isinstance(value, (np.integer, np.floating)):
            return value.item()
        elif isinstance(value, np.ndarray):
            return value.tolist()
        else:
            return value

    def _deserialize_value(self, value: Any) -> Any:
        """Deserialize MongoDB value back to pandas-compatible format.

        Args:
            value: MongoDB value to deserialize

        Returns:
            Any: Pandas-compatible value
        """
        if value is None:
            return pd.NA
        else:
            return value

    def _find_collection_for_key(self, dataset_key: str) -> Optional['pymongo.collection.Collection']:
        """Find which collection contains the given dataset key.

        Args:
            dataset_key: Dataset key to search for

        Returns:
            Optional[pymongo.collection.Collection]: Collection containing the key, or None
        """
        for collection_name in self._get_all_collection_names():
            collection = self.database[collection_name]
            if collection.count_documents({"dataset_key": dataset_key}, limit=1) > 0:
                return collection
        return None

    def _key_exists(self, dataset_key: str) -> bool:
        """Check if dataset key exists in any collection.

        Args:
            dataset_key: Dataset key to check

        Returns:
            bool: True if key exists in any collection
        """
        return self._find_collection_for_key(dataset_key) is not None

    def _delete_by_key(self, dataset_key: str):
        """Delete all data for a specific dataset key.

        Args:
            dataset_key: Dataset key to delete
        """
        for collection_name in self._get_all_collection_names():
            collection = self.database[collection_name]
            collection.delete_many({"dataset_key": dataset_key})

    def _build_delete_filter(
        self,
        dataset: Optional[DatasetType] = None,
        flag: Optional[FlagType] = None,
        config: Optional[DatasetConfigType] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Build MongoDB query filter for deletion.

        Args:
            dataset: Dataset type to filter by
            flag: Flag to filter by
            config: Configuration to filter by
            **kwargs: Additional keyword arguments to filter by

        Returns:
            Dict[str, Any]: MongoDB query filter
        """
        conditions = []

        if dataset is not None:
            conditions.append({"dataset_key": {"$regex": f"^{dataset.name}_"}})

        if flag is not None:
            conditions.append({"dataset_key": {"$regex": f"_{flag}_"}})

        if config is not None:
            config_hash = self._get_config_hash(config)
            if config_hash:
                conditions.append({"dataset_key": {"$regex": f"_config_{config_hash}_"}})

        if kwargs:
            kwargs_hash = self._get_kwargs_hash(kwargs)
            conditions.append({"dataset_key": {"$regex": f"_kwargs_{kwargs_hash}$"}})

        if conditions:
            return {"$and": conditions} if len(conditions) > 1 else conditions[0]
        else:
            return {}  # Match all documents

    def _get_all_collection_names(self) -> List[str]:
        """Get all collection names with our prefix.

        Returns:
            List[str]: List of collection names matching the prefix
        """
        all_collections = self.database.list_collection_names()
        return [c for c in all_collections if c.startswith(self.collection_prefix)]