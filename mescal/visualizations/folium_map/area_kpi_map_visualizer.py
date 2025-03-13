import folium
from tqdm import tqdm

from mescal.kpis import ValueComparisonKPI, KPI, KPICollection
from mescal.study_manager import StudyManager
from mescal.units import Units
from mescal.utils.logging import get_logger
from mescal.utils.dict_combinations import dict_combination_iterator
from mescal.utils.color_utils.segmented_colormap import SegmentedColorMap
logger = get_logger(__name__)


class AreaKPIMapVisualizer:
    def __init__(
            self,
            study_manager: StudyManager,
            print_values_on_map: bool = True,
            include_related_kpis_in_tooltip: bool = False
    ):
        self.study_manager = study_manager
        self.print_values_on_map = print_values_on_map
        self.include_related_kpis_in_tooltip = include_related_kpis_in_tooltip

    def get_feature_groups(self, kpi_collection: KPICollection, colormap: SegmentedColorMap) -> list[folium.FeatureGroup]:
        feature_groups = []
        pbar = tqdm(kpi_collection, total=kpi_collection.size, desc=f'{self.__class__.__name__}')
        with pbar:
            for kpi_group in self._get_kpi_groups(kpi_collection):
                group_name = self._get_feature_group_name(kpi_group)
                fg = folium.FeatureGroup(name=group_name, overlay=False, show=False)
                # TODO: consistent category_orders
                for kpi in kpi_group:
                    try:
                        self._add_kpi_to_feature_group(kpi, fg, colormap)
                    except Exception as e:
                        logger.warning(f'Exception while trying to add KPI {kpi.name} to FeatureGroup {group_name}: {e}')
                    pbar.update(1)
                feature_groups.append(fg)
        return feature_groups

    def _get_kpi_groups(self, kpi_collection: KPICollection) -> list[KPICollection]:
        attribute_sets = kpi_collection.get_all_kpi_attributes_and_value_sets()
        relevant_attribute_sets = {
            k: v
            for k, v in attribute_sets.items()
            if k not in ['name', 'object_name', 'column_subset']  # TODO: could be something to set in __init__
        }

        groups: list[KPICollection] = []
        for group_kwargs in dict_combination_iterator(relevant_attribute_sets):
            g = kpi_collection.get_filtered_kpi_collection_by_attributes(**group_kwargs)
            if not g.empty:
                groups.append(g)
        return groups

    def _get_feature_group_name(self, kpi_group: KPICollection) -> str:
        _include = ['value_operation', 'aggregation', 'flag', 'dataset', 'unit']
        _exclude = ['variation_dataset', 'reference_dataset', 'model_flag', 'base_unit', 'dataset_type']

        attributes = kpi_group.get_in_common_kpi_attributes(primitive_values=True)
        for k in _exclude:
            attributes.pop(k, None)

        components = []
        _include += [k for k in attributes.keys() if k not in _include]
        for k in _include:
            value = attributes.pop(k, None)
            if value is not None:
                components.append(value)

        return ' '.join(components)

    def _add_kpi_to_feature_group(self, kpi: KPI, feature_group: folium.FeatureGroup, colormap: SegmentedColorMap):
        style = self._get_style_kwargs(kpi, colormap)
        highlight = self._get_highlight_kwargs(kpi, colormap)
        geojson = self._get_geojson(kpi)
        folium.GeoJson(
            geojson,
            style_function=lambda x, s=dict(style): s,
            highlight_function=lambda x, h=dict(highlight): h,
            tooltip=folium.GeoJsonTooltip(fields=['tooltip'], aliases=[''], sticky=True)
        ).add_to(feature_group)

        if self.print_values_on_map:
            self._add_kpi_value_print_to_feature_group(kpi, feature_group, style)

    def _add_kpi_value_print_to_feature_group(self, kpi: KPI, feature_group: folium.FeatureGroup, style: dict):
        icon_text = self._get_icon_text(kpi)
        surface_color = style['fillColor']
        icon_loc = self._get_icon_projection_point(kpi)
        text_color, shadow_color = self._get_contrast_and_shadow_color_for_text_on_surface(surface_color)
        icon_html = f'''
                <div style="
                    position: absolute;
                    left: 50%;
                    top: 50%;
                    transform: translate(-50%, -50%);
                    text-align: center;
                    font-size: 8pt;
                    font-weight: bold;
                    color: {text_color};
                    white-space: nowrap;
                    text-shadow:
                       -0.05px -0.05px 0 {shadow_color},  
                        0.05px -0.05px 0 {shadow_color},
                       -0.05px  0.05px 0 {shadow_color},
                        0.05px  0.05px 0 {shadow_color};
                ">
                    {icon_text}
                </div>
            '''
        folium.Marker(location=icon_loc, icon=folium.DivIcon(html=icon_html), draggable=True).add_to(feature_group)

    def _get_contrast_and_shadow_color_for_text_on_surface(self, surface_color: str) -> tuple[str, str]:
        if self._is_dark(surface_color):
            return '#F2F2F2', '#3A3A3A'
        return '#3A3A3A', '#F2F2F2'

    def _get_style_kwargs(self, kpi: KPI, colormap: SegmentedColorMap) -> dict:
        return {
            'fillColor': colormap(kpi.value),
            'color': 'white',
            'weight': 1,
            'fillOpacity': 1
        }

    def _get_highlight_kwargs(self, kpi: KPI, colormap: SegmentedColorMap) -> dict:
        highlight = self._get_style_kwargs(kpi, colormap)
        highlight['weight'] = 3
        highlight['fillOpacity'] = 0.8
        return highlight

    def _get_tooltip(self, kpi: KPI) -> str:
        kpi_name = kpi.get_kpi_name_with_dataset_name()
        kpi_quantity = Units.get_quantity_in_pretty_unit(kpi.quantity)
        kpi_text = Units.get_pretty_text_for_quantity(kpi_quantity, thousands_separator=' ')
        html = '<table style="border-collapse: collapse;">\n'
        html += f'  <tr><td style="padding: 4px 8px;"><strong>{kpi_name}</strong></td><td style="text-align: right; padding: 4px 8px;">{kpi_text}</td></tr>\n'

        if not self.include_related_kpis_in_tooltip:
            html += '</table>'
            return html

        related_groups = self._get_related_kpi_groups(kpi)
        if all(g.empty for g in related_groups.values()):
            html += '</table>'
            return html
        for name, group in related_groups.items():
            if group.empty:
                continue
            html += "<tr><p>&nbsp;</p></tr>"
            html += f'  <tr><th colspan="2" style="text-align: left; padding: 8px;">{name}</th></tr>\n'
            for related_kpi in group:
                related_kpi_name = related_kpi.get_kpi_name_with_dataset_name()
                related_kpi_quantity = Units.get_quantity_in_pretty_unit(related_kpi.quantity)
                related_kpi_value_text = Units.get_pretty_text_for_quantity(
                    related_kpi_quantity,
                    thousands_separator=' ',
                )
                html += f'  <tr><td style="padding: 4px 8px;">{related_kpi_name}</td><td style="text-align: right; padding: 4px 8px;">{related_kpi_value_text}</td></tr>\n'
        html += '<br><p>&nbsp;</p></table>'
        return html

    def _get_related_kpi_groups(self, kpi: KPI) -> dict[str, KPICollection]:
        groups = {
            'Different Comparisons / ValueOperations': KPICollection(),
            'Different Aggregations': KPICollection(),
            'Different Datasets': KPICollection(),
        }

        kpi_atts = kpi.attributes.as_dict(primitive_values=True)

        _must_contain = ['flag', 'aggregation']
        if any(kpi_atts.get(k, None) is None for k in _must_contain):
            return groups

        pre_filtered = self.study_manager.scen_comp.get_merged_kpi_collection()
        pre_filtered = pre_filtered.get_filtered_kpi_collection_by_attributes(
            object_name=kpi.get_attributed_object_name(),
            flag=kpi_atts['flag'],
            model_flag=kpi.get_attributed_model_flag(),
        )
        _main_kpi_is_value_op = isinstance(kpi, (ValueComparisonKPI, ArithmeticValueOperationKPI))

        for potential_relative in pre_filtered:
            pratts = potential_relative.attributes.as_dict(primitive_values=True)
            if pratts.get('dataset') == kpi_atts.get('dataset'):  # same ds
                if pratts.get('aggregation', None) == kpi_atts.get('aggregation'):  # same ds, agg
                    if pratts.get('value_operation', None) != kpi_atts.get('value_operation', None):
                        groups['Different Comparisons / ValueOperations'].add_kpi(potential_relative)
                        continue
                else:  # same ds, diff agg
                    if pratts.get('value_operation', None) is None:
                        groups['Different Aggregations'].add_kpi(potential_relative)
                        continue
                    elif pratts.get('value_operation') == kpi_atts.get('value_operation', None):
                        groups['Different Aggregations'].add_kpi(potential_relative)
                        continue
            elif pratts.get('aggregation', None) == kpi_atts.get('aggregation'):  # same agg, diff ds
                if pratts.get('value_operation', None) == kpi_atts.get('value_operation', None):
                    groups['Different Datasets'].add_kpi(potential_relative)
                    continue
                if not _main_kpi_is_value_op:
                    groups['Different Comparisons / ValueOperations'].add_kpi(potential_relative)
                    continue
        return groups

    def _get_geojson(self, kpi: KPI) -> dict:
        info = kpi.get_attributed_object_info_from_model()
        tooltip = self._get_tooltip(kpi)
        return {
            "type": "Feature",
            "geometry": info.geometry.__geo_interface__,
            "properties": {"tooltip": tooltip}
        }

    def _get_icon_text(self, kpi: KPI) -> str:
        icon_text = Units.get_pretty_text_for_quantity(kpi.quantity)
        return f'{icon_text}'

    def _get_icon_projection_point(self, kpi: KPI) -> tuple[float, float]:
        info = kpi.get_attributed_object_info_from_model()
        return info['projection_point'].coords[0][::-1]

    @staticmethod
    def _is_dark(color: str) -> bool:
        r, g, b = [int(color[i:i + 2], 16) for i in (1, 3, 5)]
        return (0.299 * r + 0.587 * g + 0.114 * b) < 160
