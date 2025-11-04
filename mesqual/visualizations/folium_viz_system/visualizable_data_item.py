from abc import ABC, abstractmethod
from typing import Any

import pandas as pd
from shapely import Point, Polygon, MultiPolygon, LineString

from mesqual.kpis import KPI, KPICollection


class VisualizableDataItem(ABC):
    """
    Abstract interface for data items that can be visualized on maps.
    
    Defines the contract for all data items that can be processed by the
    folium visualization system. Provides attribute access, tooltip data
    generation, and text representation for map elements.
    
    This abstraction enables polymorphic handling of different data sources
    (model DataFrames, KPI objects, custom data) within the visualization
    pipeline while maintaining consistent interface expectations.
    """

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    @abstractmethod
    def get_name(self) -> str:
        """Get a representative name for the object."""
        pass

    @abstractmethod
    def get_text_representation(self) -> str:
        """Get a representative text for the object."""
        pass

    @abstractmethod
    def get_tooltip_data(self) -> dict:
        """Get data for tooltip display."""
        pass

    @abstractmethod
    def get_object_attribute(self, attribute: str) -> Any:
        """Get value for styling from specified column."""
        pass

    @abstractmethod
    def object_has_attribute(self, attribute: str) -> bool:
        pass


class ModelDataItem(VisualizableDataItem):
    """
    Visualizable data item for model DataFrame rows.
    
    Wraps pandas Series objects (DataFrame rows) to provide the VisualizableDataItem
    interface. Commonly used for visualizing static model data like network
    topology, geographic boundaries, or reference datasets.
    
    Handles attribute access from DataFrame columns, provides meaningful object
    naming, and generates informative tooltips from available data.
    
    Args:
        object_data: Pandas Series representing a DataFrame row
        object_type: Optional type identifier for the object
        **kwargs: Additional attributes to set on the data item
        
    Examples:
        Typical usage in generators:
        >>> for _, row in model_df.iterrows():
        ...     data_item = ModelDataItem(row, object_type='BiddingZone')
        ...     generator.generate(data_item, feature_group)
        
        Access pattern:
        >>> data_item.get_object_attribute('geometry')  # From DataFrame column
        >>> data_item.get_name()  # Object identifier
        >>> data_item.get_tooltip_data()  # All available data for tooltip
    """
    OBJECT_NAME_COLUMNS = ['name', 'object_id', 'index', 'object_name']

    def __init__(self, object_data: pd.Series, object_type: str | Any = None, **kwargs):
        self.object_data = object_data
        self.object_id = object_data.name
        self.object_type = object_type
        self.name_attributes = self.OBJECT_NAME_COLUMNS + [self.object_type]
        super().__init__(**kwargs)

    def get_name(self) -> str:
        return str(self.object_id)

    def get_text_representation(self) -> str:
        return self.get_name()

    def get_tooltip_data(self) -> dict:
        data = {'ID': self.object_id}
        for col, value in self.object_data.items():
            if pd.notna(value):
                value_str = str(value)
                if len(value_str) > 50:
                    value_str = value_str[:47] + "..."
                data[col] = value_str
        return data

    def get_object_attribute(self, attribute: str) -> Any:
        if (attribute in self.name_attributes) and (attribute not in self.object_data):
            return self.get_name()
        return self.object_data.get(attribute)

    def object_has_attribute(self, attribute: str) -> bool:
        return (attribute in self.object_data) or (attribute in self.name_attributes)


class KPIDataItem(VisualizableDataItem):
    """
    Visualizable data item for KPI objects.
    
    Wraps MESQUAL KPI objects to provide the VisualizableDataItem interface.
    Combines KPI values with associated model object information for rich
    map visualization of computed energy system metrics.
    
    Handles attribute access from both KPI values and underlying model objects,
    provides formatted value representations, and generates enhanced tooltips
    showing both KPI information and object details.
    
    Args:
        kpi: MESQUAL KPI object with computed value and metadata
        kpi_collection: Optional KPI collection for context
        **kwargs: Additional attributes to set on the data item
        
    Examples:
        Typical usage in visualizers:
        >>> for kpi in kpi_collection:
        ...     data_item = KPIDataItem(kpi, kpi_collection)
        ...     generator.generate(data_item, feature_group)
        
        Access patterns:
        >>> data_item.kpi.value  # Direct KPI value access
        >>> data_item.get_object_attribute('geometry')  # From model object
        >>> data_item.get_object_attribute('kpi_value')  # Alias for KPI value
        >>> data_item.get_text_representation()  # Formatted value string
    """

    KPI_VALUE_COLUMNS = ['kpi_value', 'value', 'kpi']

    def __init__(self, kpi: KPI, kpi_collection: KPICollection = None, **kwargs):
        self.kpi = kpi
        self.kpi_collection = kpi_collection
        self._object_info = kpi.get_object_info_from_model()
        self._model_item = ModelDataItem(self._object_info)
        super().__init__(**kwargs)

    def get_name(self) -> str:
        return str(self.kpi.name)

    def get_text_representation(self, converter: 'QuantityToTextConverter' = None) -> str:
        """
        Get formatted text representation of the KPI value.

        Args:
            converter: Optional QuantityToTextConverter for custom formatting

        Returns:
            Formatted string representation of the KPI value
        """
        from mesqual.units import Units, QuantityToTextConverter

        if converter is not None:
            return converter.convert(self.kpi.quantity)

        # Default behavior: auto-select pretty unit
        q = Units.get_quantity_in_pretty_unit(self.kpi.quantity)
        text = Units.get_pretty_text_for_quantity(q, include_unit=False)
        return text

    def get_tooltip_data(self) -> dict:
        kpi_data = {
            'KPI': self.kpi.get_kpi_name_with_dataset_name(),
            'Value': str(self.kpi.quantity),
        }
        _MAX_MODEL_DATA_LEN = 3
        model_data = self._model_item.get_tooltip_data()
        model_data = dict(list(model_data.items())[:_MAX_MODEL_DATA_LEN])
        return {**kpi_data, **model_data}

    def get_object_attribute(self, attribute: str) -> Any:
        if (attribute in self.KPI_VALUE_COLUMNS) and (not self._model_item.object_has_attribute(attribute)):
            return self.kpi.value
        return self._model_item.get_object_attribute(attribute)

    def object_has_attribute(self, attribute: str) -> bool:
        if attribute in self.KPI_VALUE_COLUMNS:
            return True
        return self._model_item.object_has_attribute(attribute)
