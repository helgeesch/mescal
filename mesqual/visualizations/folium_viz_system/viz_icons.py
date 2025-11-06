from abc import ABC

import folium

from mesqual.visualizations.folium_viz_system.base_viz_system import FoliumObjectGenerator, FeatureResolver
from mesqual.visualizations.folium_viz_system.visualizable_data_item import VisualizableDataItem


class PELIMINARY_IconGenerator(FoliumObjectGenerator, ABC):
    """Generates folium Marker or CircleMarker objects for point geometries."""

    def generate(self, data_item: VisualizableDataItem, feature_group: folium.FeatureGroup) -> None:
        try:
            location = data_item.get_location()
        except ValueError:
            return

        style = self.feature_resolver.resolve_feature(data_item)

        marker_kwargs = {'location': location, 'tooltip': tooltip}
        if popup:
            marker_kwargs['popup'] = folium.Popup(popup, max_width=300)

        icon = self.icon_generator.generate_icon(data_item, style)
        folium.Marker(
            icon=icon,
            **marker_kwargs
        ).add_to(feature_group)

        folium.Marker(**marker_kwargs).add_to(feature_group)
