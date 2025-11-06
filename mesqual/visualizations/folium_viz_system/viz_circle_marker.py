from dataclasses import dataclass
from typing import Type, Any

from shapely import Point
import folium

from mesqual.visualizations.folium_viz_system.visualizable_data_item import VisualizableDataItem
from mesqual.visualizations.folium_viz_system.base_viz_system import (
    ResolvedFeature,
    FeatureResolver,
    PropertyMapper,
    FoliumObjectGenerator,
)


@dataclass
class ResolvedCircleMarkerFeature(ResolvedFeature):
    """Specialized style container for circle marker visualizations."""

    @property
    def location(self) -> Point:
        return self.get('location')

    @property
    def fill_color(self) -> str:
        return self.get('fill_color')

    @property
    def border_color(self) -> str:
        return self.get('border_color')

    @property
    def radius(self) -> float:
        return self.get('radius')

    @property
    def border_width(self) -> float:
        return self.get('border_width')

    @property
    def fill_opacity(self) -> float:
        return self.get('fill_opacity')


class CircleMarkerFeatureResolver(FeatureResolver[ResolvedCircleMarkerFeature]):
    def __init__(
            self,
            fill_color: PropertyMapper | str = '#D9D9D9',
            border_color: PropertyMapper | str = 'white',
            radius: PropertyMapper | float = 8.0,
            border_width: PropertyMapper | float = 1.0,
            fill_opacity: PropertyMapper | float = 1.0,
            tooltip: PropertyMapper | str | bool = True,
            popup: PropertyMapper | folium.Popup | bool = False,
            location: PropertyMapper | Point = None,
            **property_mappers: PropertyMapper | Any,
    ):
        mappers = dict(
            fill_color=fill_color,
            border_color=border_color,
            radius=radius,
            border_width=border_width,
            fill_opacity=fill_opacity,
            tooltip=tooltip,
            popup=popup,
            location=self._explicit_or_fallback(location, self._default_location_mapper()),
            **property_mappers
        )
        super().__init__(feature_type=ResolvedCircleMarkerFeature, **mappers)


class CircleMarkerGenerator(FoliumObjectGenerator[CircleMarkerFeatureResolver]):
    """Generates folium CircleMarker objects for point geometries."""

    def _feature_resolver_type(self) -> Type[CircleMarkerFeatureResolver]:
        return CircleMarkerFeatureResolver

    def generate(self, data_item: VisualizableDataItem, feature_group: folium.FeatureGroup) -> None:
        style = self.feature_resolver.resolve_feature(data_item)
        if style.location is None:
            return

        folium.CircleMarker(
            location=(style.location.y, style.location.x),
            tooltip=style.tooltip,
            popup=style.popup,
            radius=style.radius,
            color=style.border_color,
            fillColor=style.fill_color,
            fillOpacity=style.fill_opacity,
            weight=style.border_width,
        ).add_to(feature_group)


if __name__ == '__main__':
    import os
    import webbrowser
    import pandas as pd
    from shapely.geometry import Point
    import folium

    from mesqual.visualizations.value_mapping_system import (
        SegmentedContinuousColorscale,
        SegmentedContinuousInputToContinuousOutputMapping,
    )

    node_df = pd.DataFrame({
        'location': [Point(7.0, 50.0), Point(8.0, 50.1), Point(9.0, 50.2)],
        'value': [5, 15, 30]
    }, index=['point1', 'point2', 'point3'])

    m = folium.Map(location=[50.1, 8.0], zoom_start=7, tiles='CartoDB Positron')

    color_map = SegmentedContinuousColorscale.single_segment_autoscale_factory_from_array(
        values=node_df['value'].values,
        colorscale=['blue', 'lime', 'red']
    )

    size_map = SegmentedContinuousInputToContinuousOutputMapping.single_segment_autoscale_factory_from_array(
        values=node_df['value'].values,
        output_range=(5, 15)
    )

    marker_generator = CircleMarkerGenerator(
        feature_resolver=CircleMarkerFeatureResolver(
            fill_color=PropertyMapper.from_item_attr('value', color_map),
            radius=PropertyMapper.from_item_attr('value', size_map),
            fill_opacity=0.9,
            border_color='black',
            border_width=1
        )
    )

    fg = folium.FeatureGroup(name='Circle Markers')
    marker_generator.generate_objects_for_model_df(node_df, fg)
    fg.add_to(m)

    m.add_child(folium.LayerControl())
    m.save('_tmp/map.html')
    webbrowser.open('file://' + os.path.abspath('_tmp/map.html'))
