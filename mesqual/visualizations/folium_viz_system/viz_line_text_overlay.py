from dataclasses import dataclass
from typing import Callable, Type, Any

import folium
from folium.plugins import PolyLineTextPath
from shapely import LineString

from mesqual.visualizations.folium_viz_system.visualizable_data_item import VisualizableDataItem, KPIDataItem
from mesqual.visualizations.folium_viz_system.base_viz_system import (
    ResolvedFeature,
    FeatureResolver,
    PropertyMapper,
    FoliumObjectGenerator,
)


@dataclass
class ResolvedLineTextOverlayFeature(ResolvedFeature):
    @property
    def geometry(self) -> LineString:
        return self.get('geometry')

    @property
    def text_offset(self) -> float:
        offset = self.get('text_offset')
        if offset is None:
            return self.font_size / 2
        return offset

    @property
    def text_orientation(self) -> int:
        return self.get('text_orientation', 0)

    @property
    def text_repeat(self) -> bool:
        return self.get('text_repeat', False)

    @property
    def text_center(self) -> bool:
        return self.get('text_center', False)

    @property
    def text_below(self) -> bool:
        return self.get('text_below', False)

    @property
    def font_weight(self) -> str:
        return self.get('font_weight', 'bold')

    @property
    def font_size(self) -> int:
        return self.get('font_size', 12)

    @property
    def font_color(self) -> str:
        return self.get('font_color', '#000000')

    @property
    def reverse_path_direction(self) -> bool:
        return self.get('reverse_path_direction', False)


class LineTextOverlayFeatureResolver(FeatureResolver[ResolvedLineTextOverlayFeature]):
    def __init__(
            self,
            text_offset: PropertyMapper | float = None,
            text_orientation: PropertyMapper | int = 0,
            text_repeat: PropertyMapper | bool = False,
            text_center: PropertyMapper | bool = False,
            text_below: PropertyMapper | bool = False,
            font_weight: PropertyMapper | str = 'bold',
            font_size: PropertyMapper | int = 12,
            font_color: PropertyMapper | str = '#000000',
            reverse_path_direction: PropertyMapper | bool = False,
            text_print_content: PropertyMapper | str = True,
            tooltip: PropertyMapper | str | bool = True,
            popup: PropertyMapper | folium.Popup | bool = False,
            geometry: PropertyMapper | LineString = None,
            **property_mappers: PropertyMapper | Any,
    ):
        mappers = dict(
            text_offset=text_offset,
            text_orientation=text_orientation,
            text_repeat=text_repeat,
            text_center=text_center,
            text_below=text_below,
            font_weight=font_weight,
            font_size=font_size,
            font_color=font_color,
            reverse_path_direction=reverse_path_direction,
            text_print_content=text_print_content,
            tooltip=tooltip,
            popup=popup,
            geometry=self._explicit_or_fallback(geometry, self._default_line_string_mapper()),
            **property_mappers
        )
        super().__init__(feature_type=ResolvedLineTextOverlayFeature, **mappers)


class LineTextOverlayGenerator(FoliumObjectGenerator[LineTextOverlayFeatureResolver]):
    def _feature_resolver_type(self) -> Type[LineTextOverlayFeatureResolver]:
        return LineTextOverlayFeatureResolver

    def generate(self, data_item: VisualizableDataItem, feature_group: folium.FeatureGroup) -> None:
        style = self.feature_resolver.resolve_feature(data_item)
        if not isinstance(style.geometry, LineString):
            return

        if not style.text_print_content:
            return

        coordinates = [(lat, lon) for lon, lat in style.geometry.coords]

        if style.reverse_path_direction:
            coordinates = coordinates[::-1]

        invisible_line = folium.PolyLine(
            locations=coordinates,
            color=None,
            opacity=0.0,
        )
        invisible_line.add_to(feature_group)

        PolyLineTextPath(
            invisible_line,
            text=style.text_print_content,
            repeat=style.text_repeat,
            center=style.text_center,
            below=style.text_below,
            orientation=style.text_orientation,
            offset=style.text_offset,
            attributes={
                'font-weight': style.font_weight,
                'font-size': str(style.font_size),
                'fill': style.font_color,
            }
        ).add_to(feature_group)


if __name__ == '__main__':
    import os
    import webbrowser
    import pandas as pd
    from shapely.geometry import Polygon, LineString
    import folium

    from mesqual.visualizations.value_mapping_system import (
        SegmentedContinuousColorscale,
        SegmentedContinuousOpacityMapping,
    )
    from mesqual.visualizations.folium_viz_system.viz_lines import LineGenerator, LineFeatureResolver

    line_df = pd.DataFrame({
        'geometry': [
            LineString([(7.0, 50.0), (7.5, 52.0)]),
            LineString([(8.0, 50.0), (8.5, 52.0)]),
            LineString([(9.0, 50.0), (9.5, 52.0)]),
        ],
        'flow': [10, -20, 30]
    }, index=['line1', 'line2', 'line3.1'])

    m = folium.Map(location=[50.25, 8.0], zoom_start=7, tiles='CartoDB Positron')

    color_map = SegmentedContinuousColorscale.single_segment_autoscale_factory_from_array(
        values=line_df['flow'].values,
        colorscale=['green', 'blue', 'red']
    )

    opacity_map = SegmentedContinuousOpacityMapping.single_segment_autoscale_factory_from_array(
        values=line_df['flow'].values,
        output_range=(0.4, 0.9)
    )

    line_generator = LineGenerator(
        feature_resolver=LineFeatureResolver(
            line_color=PropertyMapper.from_item_attr('flow', color_map),
            line_opacity=PropertyMapper.from_item_attr('flow', opacity_map),
            line_width=10,
            tooltip=False,
            popup=True,
        ),
        offset_increment=15,
    )

    line_text_generator = LineTextOverlayGenerator(
        feature_resolver=LineTextOverlayFeatureResolver(
            text_print_content=PropertyMapper.from_item_attr('flow', lambda x: f"\u25B6 {abs(x):.0f} MW \u25B6"),
            reverse_path_direction=PropertyMapper.from_item_attr('flow', lambda x: x < 0),
            text_center=True,
        ),
    )

    fg = folium.FeatureGroup(name='Test LineTextOverlay')
    line_generator.generate_objects_for_model_df(line_df, fg)
    line_text_generator.generate_objects_for_model_df(line_df, fg)
    fg.add_to(m)

    m.add_child(folium.LayerControl())
    m.save('_tmp/map.html')
    webbrowser.open('file://' + os.path.abspath('_tmp/map.html'))
