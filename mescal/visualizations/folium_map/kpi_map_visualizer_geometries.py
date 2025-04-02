from abc import ABC, abstractmethod
import folium

from mescal.kpis import KPI
from mescal.study_manager import StudyManager
from mescal.visualizations.styling.segmented_colormap import SegmentedColorMapLegend
from mescal.visualizations.folium_map.kpi_map_visualizer_base import KPIToMapVisualizerBase


class GeometryKPIMapVisualizer(KPIToMapVisualizerBase, ABC):
    def _add_kpi_to_feature_group(self, kpi: KPI, feature_group: folium.FeatureGroup):
        style = self._get_style_kwargs(kpi)
        highlight = self._get_highlight_kwargs(kpi)
        geojson = self._get_geojson(kpi)
        folium.GeoJson(
            geojson,
            style_function=lambda x, s=dict(style): s,
            highlight_function=lambda x, h=dict(highlight): h,
            tooltip=folium.GeoJsonTooltip(fields=['tooltip'], aliases=[''], sticky=True)
        ).add_to(feature_group)

        if self.print_values_on_map:
            self._add_kpi_value_print_to_feature_group(kpi, feature_group, style)

    def _get_geojson(self, kpi: KPI) -> dict:
        info = kpi.get_attributed_object_info_from_model()
        tooltip = self._get_tooltip(kpi)
        return {
            "type": "Feature",
            "geometry": info.geometry.__geo_interface__,
            "properties": {"tooltip": tooltip}
        }

    @abstractmethod
    def _get_style_kwargs(self, kpi: KPI) -> dict:
        return {
            'color': '#AABBCC',
            'weight': 1.0,
            'opacity': 1.0
        }

    @abstractmethod
    def _get_highlight_kwargs(self, kpi: KPI) -> dict:
        highlight = self._get_style_kwargs(kpi)
        highlight['weight'] = highlight['weight'] * 1.5  # Make the line thicker on highlight
        highlight['opacity'] = 0.8
        return highlight


class AreaKPIMapVisualizer(GeometryKPIMapVisualizer):
    def __init__(
            self,
            study_manager: StudyManager,
            colormap: SegmentedColorMapLegend,
            print_values_on_map: bool = True,
            include_related_kpis_in_tooltip: bool = False,
    ):
        self.colormap = colormap
        super().__init__(study_manager, print_values_on_map, include_related_kpis_in_tooltip)

    def _get_style_kwargs(self, kpi: KPI) -> dict:
        return {
            'fillColor': self.colormap(kpi.value),
            'color': 'white',
            'weight': 1,
            'fillOpacity': 1
        }

    def _get_highlight_kwargs(self, kpi: KPI) -> dict:
        highlight = self._get_style_kwargs(kpi)
        highlight['weight'] = 3
        highlight['fillOpacity'] = 0.8
        return highlight


class LineKPIMapVisualizer(GeometryKPIMapVisualizer):
    def __init__(
            self,
            study_manager: StudyManager,
            colormap: SegmentedColorMapLegend,
            widthmap: LineWidthMap | float = 3.0,
            print_values_on_map: bool = True,
            include_related_kpis_in_tooltip: bool = False,
    ):
        super().__init__(study_manager, print_values_on_map, include_related_kpis_in_tooltip)
        self.colormap = colormap
        self.widthmap = widthmap if isinstance(widthmap, LineWidthMap) else lambda x: widthmap

    def _get_style_kwargs(self, kpi: KPI) -> dict:
        return {
            'color': self.colormap(kpi.value),
            'weight': self.widthmap(kpi.value),
            'opacity': 1.0
        }

    def _get_highlight_kwargs(self, kpi: KPI) -> dict:
        highlight = self._get_style_kwargs(kpi)
        highlight['weight'] = highlight['weight'] * 1.5  # Make the line thicker on highlight
        highlight['opacity'] = 0.8
        return highlight