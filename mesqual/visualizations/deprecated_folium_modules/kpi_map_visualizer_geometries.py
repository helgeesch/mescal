from abc import ABC, abstractmethod
import folium
from shapely.geometry import LineString, Point

from mesqual._kpis_deprecated import KPI
from mesqual.study_manager import StudyManager
from mesqual.visualizations.deprecated_styling.segmented_colormap import SegmentedColorMapLegend
from mesqual.visualizations.deprecated_styling.segmented_line_width_map import SegmentedLineWidthMapLegend
from mesqual.visualizations.deprecated_styling.segmented_opacity_map import SegmentedOpacityMapLegend
from mesqual.visualizations.deprecated_folium_modules.kpi_map_visualizer_base import KPIToMapVisualizerBase


class GeometryKPIMapVisualizer(KPIToMapVisualizerBase, ABC):
    def __init__(
            self,
            study_manager: StudyManager,
            colormap: SegmentedColorMapLegend | str = '#D9D9D9',
            widthmap: SegmentedLineWidthMapLegend | float = 3.0,
            opacitymap: SegmentedOpacityMapLegend | float = 1.0,
            print_values_on_map: bool = True,
            include_related_kpis_in_tooltip: bool = False,
    ):
        super().__init__(study_manager, print_values_on_map, include_related_kpis_in_tooltip)
        self.colormap = colormap if isinstance(colormap, SegmentedColorMapLegend) else lambda x: colormap
        self.widthmap = widthmap if isinstance(widthmap, SegmentedLineWidthMapLegend) else lambda x: widthmap
        self.opacitymap = opacitymap if isinstance(opacitymap, SegmentedOpacityMapLegend) else lambda x: opacitymap


class AreaKPIMapVisualizer(GeometryKPIMapVisualizer):

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
            self._add_kpi_value_print_to_feature_group(kpi, feature_group, style['fillColor'])

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
            'weight': self.widthmap(kpi.value),
            'fillOpacity': self.opacitymap(kpi.value)
        }

    def _get_highlight_kwargs(self, kpi: KPI) -> dict:
        highlight = self._get_style_kwargs(kpi)
        highlight['weight'] = self.widthmap(kpi.value) * 1.5
        highlight['fillOpacity'] = min([self.opacitymap(kpi.value) * 1.5, 1])
        return highlight


class LineKPIMapVisualizer(GeometryKPIMapVisualizer):
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
            opacity=self.opacitymap(kpi.value),
            tooltip=self._get_tooltip(kpi),
        ).add_to(feature_group)

        if self.print_values_on_map:
            self._add_kpi_value_print_to_feature_group(kpi, feature_group)
