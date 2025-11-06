from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Union, Callable, Any, Generic, List, Type, Dict, TYPE_CHECKING

from shapely import Point, Polygon, MultiPolygon, LineString, MultiLineString
import pandas as pd
import folium

from mesqual.typevars import ResolvedFeatureType, FeatureResolverType
from mesqual.utils.pandas_utils import add_index_as_column
from mesqual.visualizations.folium_viz_system.visualizable_data_item import VisualizableDataItem, ModelDataItem, KPIDataItem

if TYPE_CHECKING:
    from mesqual.kpis import KPI, KPICollection


class PropertyMapper:
    """
    Maps data item attributes to visual properties for folium map visualization.

    Core abstraction for converting model data, KPI values, or static values into
    visual properties (colors, sizes, positions, etc.) for map elements. Used throughout
    the folium visualization system to create dynamic, data-driven map styling.

    The PropertyMapper encapsulates a transformation function that takes a VisualizableDataItem
    and returns a styled value. This enables powerful declarative map styling where
    visual properties are automatically computed from underlying data.

    Examples:
        Basic color mapping from KPI values:
        >>> color_mapper = PropertyMapper.from_kpi_value(lambda v: 'red' if v > 0 else 'blue')

        Size mapping from model attributes:
        >>> size_mapper = PropertyMapper.from_item_attr('capacity', lambda c: c / 100)

        Static styling:
        >>> border_mapper = PropertyMapper.from_static_value('#000000')

        Complex conditional styling:
        >>> def complex_color(data_item: KPIDataItem):
        ...     kpi_val = data_item.kpi.value
        ...     threshold = data_item.get_object_attribute('threshold')
        ...     return 'green' if kpi_val > threshold else 'red'
        >>> mapper = PropertyMapper(complex_color)
    """
    def __init__(self, mapping: Callable[[VisualizableDataItem], Any]):
        self.mapping = mapping

    def map_data_item(self, data_item: VisualizableDataItem) -> Any:
        """
        Map data item to property value.

        Args:
            data_item: Data item to map

        Returns:
            Mapped property value
        """
        return self.mapping(data_item)

    @classmethod
    def from_static_value(cls, value: Any) -> 'PropertyMapper':
        """
        Create mapper that returns the same value for all data items.

        Used for consistent styling across all map elements (e.g., all borders
        the same color, all markers the same size).

        Args:
            value: Static value to return for all data items

        Returns:
            PropertyMapper that always returns the static value

        Examples:
            >>> border_color = PropertyMapper.from_static_value('#FFFFFF')
            >>> opacity = PropertyMapper.from_static_value(0.8)
        """
        return cls(lambda data_item: value)

    @classmethod
    def from_item_attr(
            cls,
            attribute: str,
            mapping: Callable[[Any], Any] = None,
    ) -> 'PropertyMapper':
        """
        Create mapper from model/object attribute with optional transformation.

        Extracts values from model data attributes (geometry, capacity, name, etc.)
        and optionally applies a transformation function. The attribute is resolved
        from the underlying model DataFrame or object data.

        Args:
            attribute: Name of the attribute to extract (e.g., 'geometry', 'capacity'), must be an entry in the object pd.Series
            mapping: Optional transformation function to apply to the attribute value

        Returns:
            PropertyMapper that extracts and optionally transforms the attribute

        Examples:
            >>> # Direct attribute access
            >>> geom_mapper = PropertyMapper.from_item_attr('geometry')
            >>>
            >>> # With color scale transformation
            >>> color_scale = SegmentedContinuousColorscale(...)
            >>> color_mapper = PropertyMapper.from_item_attr('capacity', color_scale)
            >>>
            >>> # With custom transformation
            >>> size_mapper = PropertyMapper.from_item_attr('power_mw',
            ...                                           lambda mw: min(max(mw/10, 5), 50))
        """
        if mapping is None:
            mapping = lambda x: x
        return cls(lambda data_item: mapping(data_item.get_object_attribute(attribute)))

    @classmethod
    def from_kpi_value(cls, mapping: Callable[[Any], Any]) -> 'PropertyMapper':
        """
        Create mapper from KPI values with transformation function.

        Specifically designed for KPIDataItem objects, extracts the computed KPI value
        and applies a transformation. Used for styling based on energy system metrics
        like power flows, prices, or emissions.

        Args:
            mapping: Transformation function applied to the KPI value

        Returns:
            PropertyMapper that transforms KPI values

        Examples:
            >>> # Color mapping for power flows
            >>> flow_colors = PropertyMapper.from_kpi_value(
            ...     lambda v: 'red' if v > 1000 else 'green'
            ... )
            >>>
            >>> # Size mapping for prices
            >>> price_sizes = PropertyMapper.from_kpi_value(
            ...     lambda p: min(max(p * 2, 10), 100)
            ... )
            >>>
            >>> # Using value mapping system
            >>> colorscale = SegmentedContinuousColorscale(...)
            >>> colors = PropertyMapper.from_kpi_value(colorscale)
        """
        return cls(lambda data_item: mapping(data_item.kpi.value))


@dataclass
class ResolvedFeature:
    """Container for resolved feature properties."""
    properties: dict = field(default_factory=dict)
    tooltip: str = None
    popup: folium.Popup = None
    text_print_content: str = None

    def get(self, property: str, default=None):
        return self.properties.get(property, default)

    def __getitem__(self, key):
        return self.properties[key]

    def __setitem__(self, key, value):
        self.properties[key] = value

    def __contains__(self, key):
        return key in self.properties

    def to_dict(self) -> dict[str, Any]:
        out = dict(self.properties)
        for name in dir(self.__class__):
            attr = getattr(self.__class__, name)
            if isinstance(attr, property):
                out[name] = getattr(self, name)
        return out


class FeatureResolver(Generic[ResolvedFeatureType]):
    """
    Resolves visual feature properties from data items using PropertyMappers.

    Central orchestrator for map element styling that takes a VisualizableDataItem
    and a collection of PropertyMappers, then produces a ResolvedFeature containing
    all computed visual properties. Handles default values, tooltip generation,
    and property normalization.

    The FeatureResolver acts as a bridge between data and visualization, converting
    raw data items into styled features ready for folium map rendering. It supports
    automatic tooltip/popup generation and flexible property mapping.

    Type Parameters:
        ResolvedFeatureType: The specific resolved feature type (e.g., ResolvedAreaFeature)

    Examples:
        >>> resolver = AreaFeatureResolver(
        ...     fill_color=PropertyMapper.from_kpi_value(color_scale),
        ...     fill_opacity=PropertyMapper.from_static_value(0.8),
        ...     tooltip=True  # Auto-generate tooltip
        ... )
        >>> resolved = resolver.resolve_feature(kpi_data_item)
    """
    def __init__(
        self,
        feature_type: Type[ResolvedFeatureType] = None,
        **property_mappers: PropertyMapper | Any
    ):
        self.feature_type: Type[ResolvedFeatureType] = feature_type or ResolvedFeatureType.__constraints__[0]

        _defaults_if_true = dict(
            tooltip=self._default_tooltip_generator,
            popup=self._default_popup_generator,
            text_print_content=self._default_text_print_generator,
        )
        for k, mapper_factory in _defaults_if_true.items():
            if property_mappers.get(k, None) is True:
                property_mappers[k] = mapper_factory()
            elif property_mappers.get(k, None) is False:
                property_mappers[k] = None

        self.property_mappers: dict[str, PropertyMapper] = self._normalize_property_mappers(property_mappers)

    def resolve_feature(self, data_item: VisualizableDataItem) -> ResolvedFeatureType:
        """
        Resolve feature properties from data item.

        Args:
            data_item: Data item to resolve

        Returns:
            Resolved feature with all computed properties
        """
        resolved = self.feature_type()
        for prop, mapper in self.property_mappers.items():
            resolved[prop] = mapper.map_data_item(data_item)
            if prop in ['tooltip', 'popup', 'text_print_content']:
                setattr(resolved, prop, mapper.map_data_item(data_item))
        return resolved

    @staticmethod
    def _normalize_property_mappers(mappers: dict[str, PropertyMapper | Any]) -> dict[str, PropertyMapper]:
        return {
            key: mapper if isinstance(mapper, PropertyMapper) else PropertyMapper.from_static_value(mapper)
            for key, mapper in mappers.items()
        }

    @staticmethod
    def _explicit_or_fallback(explicit: Any, fallback: PropertyMapper = None) -> PropertyMapper:
        if explicit is not None:
            return explicit
        return fallback

    @staticmethod
    def _default_tooltip_generator() -> PropertyMapper:
        """
        Create default tooltip generator showing data item information.

        Returns:
            PropertyMapper that generates HTML table tooltips with data item attributes
        """

        def get_tooltip(data_item: VisualizableDataItem) -> str:
            tooltip_data = data_item.get_tooltip_data()

            html = '<table style="border-collapse: collapse;">\n'
            for key, value in tooltip_data.items():
                html += f'  <tr><td style="padding: 4px 8px;"><strong>{key}</strong></td>' \
                        f'<td style="text-align: right; padding: 4px 8px;">{value}</td></tr>\n'
            html += '</table>'

            return html

        return PropertyMapper(get_tooltip)

    @staticmethod
    def _default_popup_generator() -> PropertyMapper:
        """
        Create default popup generator with formatted data item information.

        Returns:
            PropertyMapper that generates folium.Popup objects with data tables
        """

        def get_popup(data_item: VisualizableDataItem) -> folium.Popup:
            tooltip_data = data_item.get_tooltip_data()

            html = '<table style="border-collapse: collapse;">\n'
            for key, value in tooltip_data.items():
                html += f'  <tr><td style="padding: 4px 8px;"><strong>{key}</strong></td>' \
                        f'<td style="text-align: right; padding: 4px 8px;">{value}</td></tr>\n'
            html += '</table>'

            return folium.Popup(html, max_width=300)

        return PropertyMapper(get_popup)

    @staticmethod
    def _default_text_print_generator() -> PropertyMapper:
        """
        Create default text content generator for overlay labels.

        Returns:
            PropertyMapper that returns data item text representation
        """
        def get_text(data_item: VisualizableDataItem) -> str:
            return data_item.get_text_representation()

        return PropertyMapper(get_text)

    @staticmethod
    def _default_geometry_mapper() -> PropertyMapper:
        """
        Create default geometry mapper that extracts geometric objects.

        Returns:
            PropertyMapper that extracts 'geometry' attribute from data items
        """

        def get_geometry(data_item: VisualizableDataItem) -> Polygon | None:
            if data_item.object_has_attribute('geometry'):
                return data_item.get_object_attribute('geometry')
            return None

        return PropertyMapper(get_geometry)

    @staticmethod
    def _default_location_mapper() -> PropertyMapper:
        """
        Create smart location mapper with multiple fallback strategies.

        Attempts to extract Point locations from data items using various
        attribute names and geometric calculations. Handles common location
        attribute names and derives locations from complex geometries.

        Returns:
            PropertyMapper that intelligently extracts Point locations
        """

        def get_location(data_item: VisualizableDataItem) -> Point | None:
            for k in ['location', 'projection_point', 'centroid', 'midpoint']:
                if data_item.object_has_attribute(k):
                    location = data_item.get_object_attribute(k)
                    if isinstance(location, Point):
                        return location

            for lat, lon in [('lat', 'lon'), ('latitude', 'longitude')]:
                if data_item.object_has_attribute(lat) and data_item.object_has_attribute(lon):
                    lat_value = data_item.get_object_attribute(lat)
                    lon_value = data_item.get_object_attribute(lon)
                    if all(isinstance(v, (int, float)) for v in [lat_value, lon_value]):
                        return Point([lon_value, lat_value])

            if data_item.object_has_attribute('geometry'):
                geometry = data_item.get_object_attribute('geometry')
                if isinstance(geometry, Point):
                    return geometry
                elif isinstance(geometry, (Polygon, MultiPolygon)):
                    return geometry.representative_point()
                elif isinstance(geometry, (LineString, MultiLineString)):
                    return geometry.interpolate(0.5, normalized=True)
            return None

        return PropertyMapper(get_location)

    @staticmethod
    def _default_line_string_mapper() -> PropertyMapper:
        """
        Create default LineString geometry mapper for line visualizations.

        Returns:
            PropertyMapper that extracts LineString geometries from data items
        """

        def get_line_string(data_item: VisualizableDataItem) -> LineString | None:
            for k in ['geometry', 'line_string']:
                if data_item.object_has_attribute(k):
                    line_string = data_item.get_object_attribute(k)
                    if isinstance(line_string, (LineString, MultiLineString)):
                        return line_string
            return None

        return PropertyMapper(get_line_string)


class FoliumObjectGenerator(Generic[FeatureResolverType], ABC):
    """
    Abstract base class for generating folium map objects from data items.

    Defines the interface for converting VisualizableDataItems into folium
    map elements (areas, lines, markers, etc.). Each generator type handles
    a specific kind of map visualization and uses a corresponding FeatureResolver
    to compute visual properties.

    The generator pattern enables modular, composable map building where different
    visualization types can be combined within the same map. Generators can process
    both model DataFrames and KPI collections.

    Type Parameters:
        FeatureResolverType: The specific feature resolver type used by this generator

    Examples:
        Typical usage in map building:
        >>> area_gen = AreaGenerator(AreaFeatureResolver(fill_color=...))
        >>> line_gen = LineGenerator(LineFeatureResolver(line_color=...))
        >>>
        >>> fg = folium.FeatureGroup('My Data')
        >>> area_gen.generate_objects_for_model_df(model_df, fg)
        >>> line_gen.generate_objects_for_kpi_collection(kpi_collection, fg)
    """

    def __init__(
            self,
            feature_resolver: FeatureResolverType = None,
    ):
        self.feature_resolver: FeatureResolverType = feature_resolver or self._feature_resolver_type()()

    @abstractmethod
    def _feature_resolver_type(self) -> Type[FeatureResolverType]:
        return FeatureResolver

    @abstractmethod
    def generate(self, data_item: VisualizableDataItem, feature_group: folium.FeatureGroup) -> None:
        """
        Generate folium object and add it to the feature group.

        Args:
            data_item: Data item to visualize
            feature_group: Feature group to add to
        """
        pass

    def generate_objects_for_model_df(
            self,
            model_df: pd.DataFrame,
            feature_group: folium.FeatureGroup,
            **kwargs
    ) -> folium.FeatureGroup:
        """Add model DataFrame data to the map."""
        model_dff = add_index_as_column(model_df)
        object_type = model_dff.index.name if isinstance(model_dff.index.name, str) else None
        for _, row in model_dff.iterrows():
            data_item = ModelDataItem(row, object_type=object_type, **kwargs)
            self.generate(data_item, feature_group)
        return feature_group

    def generate_objects_for_kpi_collection(
            self,
            kpi_collection: 'KPICollection',
            feature_group: folium.FeatureGroup,
            **kwargs
    ) -> folium.FeatureGroup:
        """Add KPI data to the map."""
        for kpi in kpi_collection:
            data_item = KPIDataItem(kpi, kpi_collection, **kwargs)
            self.generate(data_item, feature_group)
        return feature_group

    def generate_object_for_single_kpi(
            self,
            kpi: 'KPI',
            feature_group: folium.FeatureGroup,
            kpi_collection: 'KPICollection' = None,
            **kwargs
    ) -> folium.FeatureGroup:
        """Add a single KPI to the map with optional context."""
        data_item = KPIDataItem(kpi, kpi_collection, **kwargs)
        self.generate(data_item, feature_group)
        return feature_group
