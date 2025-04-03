from abc import ABC, abstractmethod
import folium
from shapely.geometry import LineString, Point

from mescal.kpis import KPI
from mescal.study_manager import StudyManager
from mescal.visualizations.styling.segmented_colormap import SegmentedColorMapLegend
from mescal.visualizations.styling.segmented_line_width_map import SegmentedLineWidthMapLegend
from mescal.visualizations.folium_map.kpi_map_visualizer_base import KPIToMapVisualizerBase


class AreaKPIMapVisualizer(KPIToMapVisualizerBase):
    def __init__(
            self,
            study_manager: StudyManager,
            colormap: SegmentedColorMapLegend,
            print_values_on_map: bool = True,
            include_related_kpis_in_tooltip: bool = False,
    ):
        self.colormap = colormap
        super().__init__(study_manager, print_values_on_map, include_related_kpis_in_tooltip)

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
            self._add_kpi_value_print_to_feature_group(kpi, feature_group, style['color'])

    def _get_geojson(self, kpi: KPI) -> dict:
        info = kpi.get_attributed_object_info_from_model()
        tooltip = self._get_tooltip(kpi)
        return {
            "type": "Feature",
            "geometry": info.geometry.__geo_interface__,
            "properties": {"tooltip": tooltip}
        }

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


class LineKPIMapVisualizer(KPIToMapVisualizerBase):
    def __init__(
            self,
            study_manager: StudyManager,
            colormap: SegmentedColorMapLegend,
            widthmap: SegmentedLineWidthMapLegend | float = 3.0,
            print_values_on_map: bool = True,
            include_related_kpis_in_tooltip: bool = False,
    ):
        super().__init__(study_manager, print_values_on_map, include_related_kpis_in_tooltip)
        self.colormap = colormap
        self.widthmap = widthmap if isinstance(widthmap, SegmentedLineWidthMapLegend) else lambda x: widthmap

    def _add_kpi_to_feature_group(self, kpi: KPI, feature_group: folium.FeatureGroup):
        info = kpi.get_attributed_object_info_from_model()
        if isinstance(info.geometry, LineString):
            coordinates = [(lat, lon) for lon, lat in info.geometry.coords]
        elif isinstance(info.geometry, Point):
            coordinates = [tuple(info.geometry.coords[::-1])]
        else:
            raise NotImplementedError(f'Type {type(info.geometry)} not Implemented.')
        folium.PolyLine(
            coordinates,
            color=self.colormap(kpi.value),
            weight=self.widthmap(kpi.value),
            opacity=0.7,
            tooltip=self._get_tooltip(kpi),
        ).add_to(feature_group)

        if self.print_values_on_map:
            self._add_kpi_value_print_to_feature_group(kpi, feature_group)
