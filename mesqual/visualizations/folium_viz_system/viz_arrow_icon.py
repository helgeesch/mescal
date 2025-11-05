from dataclasses import dataclass
from typing import Type, Any, TYPE_CHECKING, Union
import base64

from shapely import Point
import folium

from mesqual.visualizations.folium_viz_system.visualizable_data_item import VisualizableDataItem
from mesqual.visualizations.folium_viz_system.base_viz_system import (
    ResolvedFeature,
    FeatureResolver,
    PropertyMapper,
    FoliumObjectGenerator,
)

if TYPE_CHECKING:
    from captain_arro import ArrowTypeEnum


@dataclass
class ResolvedArrowIconFeature(ResolvedFeature):
    """
    Resolved visual properties for animated arrow icon elements.
    
    Container for all computed styling properties of arrow icon visualizations,
    including position, orientation, colors, animation settings, and size.
    Used by ArrowIconGenerator to create folium markers with SVG arrow icons.
    """
    @property
    def location(self) -> Point:
        return self.get('location')

    @property
    def offset(self) -> tuple[float, float]:
        return self.get('offset')

    @property
    def azimuth_angle(self) -> float:
        return self.get('azimuth_angle')

    @property
    def arrow_type(self) -> 'ArrowTypeEnum':
        return self.get('arrow_type')

    @property
    def color(self) -> str:
        return self.get('color')

    @property
    def stroke_width(self) -> int:
        return self.get('stroke_width')

    @property
    def width(self) -> int:
        return self.get('width')

    @property
    def height(self) -> int:
        return self.get('height')

    @property
    def speed_in_px_per_second(self) -> float:
        return self.get('speed_in_px_per_second')

    @property
    def speed_in_duration_seconds(self) -> float:
        return self.get('speed_in_duration_seconds')

    @property
    def num_arrows(self) -> int:
        return self.get('num_arrows')

    @property
    def opacity(self) -> float:
        return self.get('opacity')

    @property
    def reverse_direction(self) -> bool:
        return self.get('reverse_direction')


class ArrowIconFeatureResolver(FeatureResolver[ResolvedArrowIconFeature]):
    """
    Resolves visual properties for animated arrow icon elements.
    
    Specialized feature resolver for arrow icon visualizations that handles point
    locations, arrow styling, animation parameters, and directional indicators.
    Commonly used for visualizing directional flows, network connections, or
    any point-based data with directional significance.
    
    Integrates with the captain_arro library to provide various arrow types
    and animation effects for dynamic flow visualization.
    
    Args:
        arrow_type: Type of arrow from ArrowTypeEnum (static value or PropertyMapper)
        color: Arrow color (static value or PropertyMapper)
        stroke_width: Arrow stroke width in pixels (static value or PropertyMapper)
        width: Arrow icon width in pixels (static value or PropertyMapper)
        height: Arrow icon height in pixels (static value or PropertyMapper)
        speed_in_px_per_second: Animation speed in pixels/second (static value or PropertyMapper)
        speed_in_duration_seconds: Animation duration in seconds (static value or PropertyMapper)
        num_arrows: Number of animated arrows (static value or PropertyMapper)
        opacity: Icon opacity 0-1 (static value or PropertyMapper)
        reverse_direction: Reverse arrow direction (static value or PropertyMapper)
        tooltip: Tooltip content (True for auto-generated, False for none)
        popup: Popup content (True/False/PropertyMapper)
        location: Point location (defaults to smart location detection)
        rotation_angle: Arrow rotation angle in degrees (static value or PropertyMapper)
        **property_mappers: Additional custom property mappings
        
    Examples:
        Basic directional flow arrows:
        >>> from captain_arro import ArrowTypeEnum
        >>> resolver = ArrowIconFeatureResolver(
        ...     arrow_type=ArrowTypeEnum.MOVING_FLOW_ARROW,
        ...     color='#FF0000',
        ...     width=50,
        ...     height=30
        ... )
        
        Data-driven flow visualization:
        >>> flow_color_scale = SegmentedContinuousColorscale(...)
        >>> size_scale = SegmentedContinuousInputToContinuousOutputMapping(...)
        >>> resolver = ArrowIconFeatureResolver(
        ...     arrow_type=PropertyMapper.from_kpi_value(
        ...         lambda v: ArrowTypeEnum.SPOTLIGHT_FLOW_ARROW if abs(v) > 100 
        ...                   else ArrowTypeEnum.BOUNCING_SPREAD_ARROW
        ...     ),
        ...     color=PropertyMapper.from_kpi_value(flow_color_scale),
        ...     width=PropertyMapper.from_kpi_value(size_scale),
        ...     reverse_direction=PropertyMapper.from_kpi_value(lambda v: v < 0),
        ...     rotation_angle=PropertyMapper.from_item_attr('azimuth_angle')
        ... )
    """
    def __init__(
            self,
            arrow_type: Union[PropertyMapper, 'ArrowTypeEnum'] = None,
            color: PropertyMapper | str = '#2563eb',
            stroke_width: PropertyMapper | int = 8,
            width: PropertyMapper | int = 60,
            height: PropertyMapper | int = 60,
            speed_in_px_per_second: PropertyMapper | float | None = 20.0,
            speed_in_duration_seconds: PropertyMapper | float | None = None,
            num_arrows: PropertyMapper | int = 3,
            opacity: PropertyMapper | float = 0.8,
            reverse_direction: PropertyMapper | bool = False,
            tooltip: PropertyMapper | str | bool = True,
            popup: PropertyMapper | folium.Popup | bool = False,
            location: PropertyMapper | Point = None,
            offset: PropertyMapper | tuple[float, float] = (0, 0),
            azimuth_angle: PropertyMapper | float = None,
            **property_mappers: PropertyMapper | Any,
    ):
        from captain_arro import ArrowTypeEnum
        mappers = dict(
            arrow_type=arrow_type or ArrowTypeEnum.MOVING_FLOW_ARROW,
            color=color,
            stroke_width=stroke_width,
            width=width,
            height=height,
            speed_in_px_per_second=speed_in_px_per_second,
            speed_in_duration_seconds=speed_in_duration_seconds,
            num_arrows=num_arrows,
            opacity=opacity,
            reverse_direction=reverse_direction,
            tooltip=tooltip,
            popup=popup,
            location=self._explicit_or_fallback(location, self._default_location_mapper()),
            offset=offset,
            azimuth_angle=self._explicit_or_fallback(azimuth_angle, self._default_rotation_angle_mapper()),
            **property_mappers
        )
        super().__init__(feature_type=ResolvedArrowIconFeature, **mappers)

    @staticmethod
    def _default_rotation_angle_mapper() -> PropertyMapper:

        def get_rotation_angle(data_item: VisualizableDataItem) -> float | None:
            for k in ['azimuth_angle', 'rotation_angle', 'projection_angle']:
                if data_item.object_has_attribute(k):
                    angle = data_item.get_object_attribute(k)
                    return angle
            return None

        return PropertyMapper(get_rotation_angle)


class ArrowIconGenerator(FoliumObjectGenerator[ArrowIconFeatureResolver]):
    """
    Generates folium Marker objects with animated SVG arrow icons.
    
    Creates interactive map markers featuring animated arrow icons from the
    captain_arro library. Handles SVG generation, base64 encoding, rotation,
    and integration with folium's marker system.
    
    Commonly used for visualizing:
    - Directional border flow directions with magnitude-based styling
    - Border price-spread with magnitude-based styling
    - Any point-based directional data with animation needs
    
    Examples:
        Power flow visualization:
        >>> from captain_arro import ArrowTypeEnum
        >>> flow_color_scale = SegmentedContinuousColorscale(...)
        >>> size_mapping = SegmentedContinuousInputToContinuousOutputMapping(...)
        >>> 
        >>> generator = ArrowIconGenerator(
        ...     ArrowIconFeatureResolver(
        ...         arrow_type=PropertyMapper.from_kpi_value(
        ...             lambda v: ArrowTypeEnum.MOVING_FLOW_ARROW if abs(v) > 50
        ...                       else ArrowTypeEnum.BOUNCING_SPREAD_ARROW
        ...         ),
        ...         color=PropertyMapper.from_kpi_value(lambda v: flow_color_scale(abs(v))),
        ...         width=PropertyMapper.from_kpi_value(lambda v: size_mapping(abs(v))),
        ...         reverse_direction=PropertyMapper.from_kpi_value(lambda v: v < 0),
        ...         rotation_angle=PropertyMapper.from_item_attr('azimuth_angle'),
        ...         speed_in_duration_seconds=4
        ...     )
        ... )
        >>> 
        >>> fg = folium.FeatureGroup('Flow Arrows')
        >>> generator.generate_objects_for_kpi_collection(flow_kpis, fg)
        >>> fg.add_to(map)
        
        Border flow indicators:
        >>> generator.generate_objects_for_model_df(border_df, feature_group)
    """
    def _feature_resolver_type(self) -> Type[ArrowIconFeatureResolver]:
        return ArrowIconFeatureResolver

    def generate(self, data_item: VisualizableDataItem, feature_group: folium.FeatureGroup, value_converter=None) -> None:
        """
        Generate and add a folium Marker with animated SVG arrow icon.
        
        Args:
            data_item: Data item containing point location and associated data
            feature_group: Folium feature group to add the arrow marker to
        """
        style = self.feature_resolver.resolve_feature(data_item, value_converter=value_converter)
        if style.location is None:
            return

        svg_content = self._generate_arrow_svg(style, data_item)
        encoded_svg = base64.b64encode(svg_content.encode()).decode()

        angle = float(style.azimuth_angle) - 90  # Folium counts clockwise from right-pointing direction; normal convention is CCW
        x_offset, y_offset = style.offset

        icon_html = f'''
            <div style="
                position: absolute;
                left: 50%;
                top: 50%;
                transform: translate(-50%, -50%) translate({x_offset}px, {-y_offset}px) rotate({angle}deg);
                opacity: {style.opacity};
            ">
                <img src="data:image/svg+xml;base64,{encoded_svg}" 
                     width="{style.width}" 
                     height="{style.height}">
                </img>
            </div>
        '''

        icon = folium.DivIcon(
            html=icon_html,
            icon_size=(style.width, style.height),
            icon_anchor=(style.width // 2, style.height // 2)
        )
        folium.Marker(
            location=(style.location.y, style.location.x),
            icon=icon,
            tooltip=style.tooltip,
            popup=style.popup
        ).add_to(feature_group)

    def _generate_arrow_svg(self, style: ResolvedArrowIconFeature, data_item: VisualizableDataItem) -> str:
        from inspect import signature
        from captain_arro import get_generator_for_arrow_type

        def safe_init(cls: type, kwargs: dict):
            init_params = signature(cls.__init__).parameters
            accepted_keys = {
                k for k in init_params
                if k != 'self' and init_params[k].kind in (
                init_params[k].POSITIONAL_OR_KEYWORD, init_params[k].KEYWORD_ONLY)
            }
            filtered_kwargs = {k: v for k, v in kwargs.items() if k in accepted_keys}
            return cls(**filtered_kwargs)

        arrow_type = style.arrow_type
        generator_class = get_generator_for_arrow_type(arrow_type)
        arrow_style_kwargs = style.to_dict()
        if 'direction' not in arrow_style_kwargs:
            arrow_style_kwargs['direction'] = 'left' if style.reverse_direction else 'right'
        return safe_init(generator_class, arrow_style_kwargs).generate_svg(unique_id=True)


if __name__ == '__main__':
    import os
    import webbrowser
    import pandas as pd
    from shapely.geometry import Point
    from mesqual.visualizations.value_mapping_system import (
        SegmentedContinuousColorscale,
        SegmentedContinuousOpacityMapping,
    )
    from captain_arro import ArrowTypeEnum

    border_model_df = pd.DataFrame({
        'projection_point': [Point(7.45, 49.15), Point(6.94, 52.22), Point(6.34, 50.38), Point(12.31, 50.25)],
        # 'azimuth_angle': [-110.0, 170.0, 180.0, -15],
        'azimuth_angle': [0, 90, 180, 270],
        'border_flow': [150.0, 75.0, 200.0, 100],
        'bing_bong': [5.0, -2.0, 8.0, -10],
    }, index=['DE-FR', 'DE-NL', 'DE-BE', 'DE-CZ'])

    m = folium.Map(location=[50.0, 9.0], zoom_start=6, tiles='Cartodb Positron')

    flow_color_mapping = SegmentedContinuousColorscale.single_segment_autoscale_factory_from_array(
        values=border_model_df['border_flow'].abs().values,
        colorscale=['red', 'blue'],
    )

    size_mapping = SegmentedContinuousOpacityMapping.single_segment_autoscale_factory_from_array(
        values=border_model_df['border_flow'].abs().values,
        output_range=(40, 80)
    )

    arrow_generator = ArrowIconGenerator(
        feature_resolver=ArrowIconFeatureResolver(
            arrow_type=PropertyMapper.from_item_attr('border_flow', lambda x: ArrowTypeEnum.SPOTLIGHT_FLOW_ARROW if x > 0 else ArrowTypeEnum.MOVING_FLOW_ARROW),
            color=PropertyMapper.from_item_attr('border_flow', lambda x: flow_color_mapping(abs(x))),
            stroke_width=4,
            azimuth_angle=PropertyMapper.from_item_attr('azimuth_angle'),
            reverse_direction=PropertyMapper.from_item_attr('border_flow', lambda x: x < 0),
            speed_in_px_per_second=None,
            speed_in_duration_seconds=2,
            width=PropertyMapper.from_item_attr('border_flow', lambda x: size_mapping(abs(x))),
            height=PropertyMapper.from_item_attr('border_flow', lambda x: size_mapping(abs(x)) / 2),
        )
    )

    fg = folium.FeatureGroup(name='Border Flows')

    arrow_generator.generate_objects_for_model_df(border_model_df, fg)

    fg.add_to(m)
    m.add_child(folium.LayerControl())
    m.save('_tmp/map.html')
    webbrowser.open('file://' + os.path.abspath('_tmp/map.html'))
