from typing import TYPE_CHECKING, List, Literal, Union

import folium

from mesqual.kpis import KPICollection, KPI
from mesqual.visualizations.folium_viz_system.base_viz_system import FoliumObjectGenerator, PropertyMapper
from mesqual.visualizations.folium_viz_system.visualizable_data_item import KPIDataItem, VisualizableDataItem
from mesqual.units import QuantityToTextConverter

if TYPE_CHECKING:
    from mesqual.study_manager import StudyManager


SHOW_OPTIONS = Literal['first', 'last', 'none']
VALUE_FORMATTING_OPTIONS = Union[Literal['per_feature_group', 'per_collection'], QuantityToTextConverter]


class KPIValueConverterResolver:
    """
    Resolves value converters for KPIs based on formatting strategy.

    Manages converter creation and caching at the feature group level,
    supporting three strategies:
    - 'per_feature_group': One converter per feature group (all KPIs in group)
    - 'per_collection': One converter per base unit across entire collection
    - Explicit QuantityToTextConverter: Use provided converter for all KPIs

    Args:
        strategy: Formatting strategy to use

    Examples:
        >>> resolver = KPIValueConverterResolver('per_feature_group')
        >>> converter = resolver.get_converter_for_kpi(kpi, group_name, kpi_group)
    """

    def __init__(self, strategy: VALUE_FORMATTING_OPTIONS):
        self.strategy = strategy
        self._group_cache = {}

    def get_converter_for_kpi(
        self,
        kpi: KPI,
        group_id: str,
        kpi_group: KPICollection
    ) -> QuantityToTextConverter:
        """
        Get converter for a specific KPI, using group-level caching.

        Args:
            kpi: KPI to get converter for
            group_id: Unique identifier for the feature group (used for caching)
            kpi_group: Full KPI group for building converters

        Returns:
            QuantityToTextConverter configured for this KPI
        """
        # Strategy 1: Explicit converter - just return it
        if isinstance(self.strategy, QuantityToTextConverter):
            return self.strategy

        # Check cache first
        if group_id not in self._group_cache:
            self._group_cache[group_id] = self._build_converters_for_group(kpi_group)

        group_converters = self._group_cache[group_id]

        # Return appropriate converter
        if isinstance(group_converters, dict):
            # per_collection: lookup by base unit
            from mesqual.units import Units
            base_unit = str(Units.get_base_unit_for_unit(kpi.quantity.units))
            return group_converters[base_unit]
        else:
            # per_feature_group: single converter for whole group
            return group_converters

    def _build_converters_for_group(
        self,
        kpi_group: KPICollection
    ) -> QuantityToTextConverter | dict[str, QuantityToTextConverter]:
        """
        Build converter(s) for a feature group.

        Returns either a single converter (per_feature_group) or a dict of
        converters keyed by base unit string (per_collection).
        """
        if self.strategy == 'per_feature_group':
            quantities = [kpi.quantity for kpi in kpi_group]
            return QuantityToTextConverter.from_quantities(
                quantities,
                thousands_separator=' '
            )

        # per_collection: group by base unit
        from mesqual.units import Units
        by_base_unit = {}
        for kpi in kpi_group:
            base_unit = str(Units.get_base_unit_for_unit(kpi.quantity.units))
            if base_unit not in by_base_unit:
                by_base_unit[base_unit] = []
            by_base_unit[base_unit].append(kpi.quantity)

        return {
            base_unit: QuantityToTextConverter.from_quantities(
                quantities,
                thousands_separator=' '
            )
            for base_unit, quantities in by_base_unit.items()
        }


class KPIGroupingManager:
    """
    Manages sophisticated KPI grouping and organization for map visualization.

    Handles the complex logic of grouping KPIs by their attributes, creating
    meaningful feature group names, and finding related KPIs for enhanced
    tooltips. Supports custom sorting orders and category hierarchies.

    The grouping system is designed to create logical visual organization
    of energy system KPIs, where related metrics (same flag, different datasets)
    are grouped together and presented with consistent naming.

    Args:
        kpi_attribute_category_orders: Custom ordering for specific attribute values
        kpi_attribute_keys_to_exclude_from_grouping: Attributes to ignore during grouping
        kpi_attribute_sort_order: Order of attributes for group sorting

    Examples:

        Custom grouping configuration:
        >>> manager = KPIGroupingManager(
        ...     kpi_attribute_category_orders={
        ...         'dataset_name': ['reference', 'scenario_1', 'scenario_2'],
        ...         'aggregation': ['sum', 'mean', 'max']
        ...     },
        ...     kpi_attribute_keys_to_exclude_from_grouping=['object_name']
        ... )
    """

    DEFAULT_EXCLUDE_FROM_GROUPING = ['name', 'object_name', 'column_subset', 'custom_name',
                                      'name_prefix', 'name_suffix', 'unit', 'target_unit']
    DEFAULT_SORT_ORDER = [
        'name_prefix', 'model_flag', 'flag', 'aggregation',
        'reference_dataset_name', 'variation_dataset_name', 'dataset_name',
        'value_comparison', 'arithmetic_operation', 'name_suffix'
    ]
    DEFAULT_INCLUDE_ATTRIBUTES = ['arithmetic_operation', 'aggregation', 'flag', 'dataset_name', 'unit']
    DEFAULT_EXCLUDE_ATTRIBUTES = ['variation_dataset_name', 'reference_dataset_name', 'model_flag',
                                   'dataset_type', 'target_unit', 'dataset_attributes']

    def __init__(
            self,
            kpi_attribute_category_orders: dict[str, list[str]] = None,
            kpi_attribute_keys_to_exclude_from_grouping: list[str] = None,
            kpi_attribute_sort_order: list[str] = None
    ):
        self.kpi_attribute_category_orders = kpi_attribute_category_orders or {}
        self.kpi_attribute_keys_to_exclude_from_grouping = (
                kpi_attribute_keys_to_exclude_from_grouping or self.DEFAULT_EXCLUDE_FROM_GROUPING.copy()
        )
        self.kpi_attribute_sort_order = (
                kpi_attribute_sort_order or self.DEFAULT_SORT_ORDER.copy()
        )

    def get_kpi_groups(self, kpi_collection: KPICollection) -> list[KPICollection]:
        """
        Group KPIs by attributes with sophisticated sorting.

        Creates logical groups of KPIs based on their attributes, excluding
        specified attributes from grouping and applying custom sort orders.

        Args:
            kpi_collection: Collection of KPIs to group

        Returns:
            List of KPICollection objects, each representing a logical group
        """
        from mesqual.utils.dict_combinations import dict_combination_iterator

        attribute_sets = kpi_collection.get_all_kpi_attributes_and_value_sets(primitive_values=True)
        relevant_attribute_sets = {
            k: v for k, v in attribute_sets.items()
            if k not in self.kpi_attribute_keys_to_exclude_from_grouping
        }

        ordered_keys = [k for k in self.kpi_attribute_sort_order if k in relevant_attribute_sets]

        # Build attribute value rankings
        attribute_value_rank: dict[str, dict[str, int]] = {}
        for attr in ordered_keys:
            existing_values = set(relevant_attribute_sets.get(attr, []))
            manual_order = [v for v in self.kpi_attribute_category_orders.get(attr, []) if v in existing_values]
            remaining = list(existing_values - set(manual_order))
            try:
                remaining = list(sorted(remaining))
            except TypeError:
                pass
            full_order = manual_order + remaining
            attribute_value_rank[attr] = {val: idx for idx, val in enumerate(full_order)}

        def sorting_index(group_kwargs: dict[str, str]) -> tuple:
            return tuple(
                attribute_value_rank[attr].get(group_kwargs.get(attr), float("inf"))
                for attr in ordered_keys
            )

        # Create and sort groups
        group_kwargs_list = list(dict_combination_iterator(relevant_attribute_sets))
        group_kwargs_list.sort(key=sorting_index)

        groups: list[KPICollection] = []
        for group_kwargs in group_kwargs_list:
            g = kpi_collection.filter(**group_kwargs)
            if not g.empty:
                groups.append(g)

        return groups

    def get_feature_group_name(self, kpi_group: KPICollection) -> str:
        """
        Generate meaningful feature group name from KPI group.

        Creates human-readable names for map feature groups based on
        common KPI attributes, prioritizing important attributes.

        Args:
            kpi_group: KPI group to generate name for

        Returns:
            Human-readable feature group name
        """
        attributes = kpi_group.get_in_common_kpi_attributes(primitive_values=True)

        for k in self.DEFAULT_EXCLUDE_ATTRIBUTES:
            attributes.pop(k, None)

        components = []
        include_attrs = self.DEFAULT_INCLUDE_ATTRIBUTES + [
            k for k in attributes.keys() if k not in self.DEFAULT_INCLUDE_ATTRIBUTES
        ]
        for k in include_attrs:
            value = attributes.pop(k, None)
            if value is not None:
                components.append(str(value))

        return ' '.join(components)

    def get_related_kpi_groups(self, kpi: KPI, study_manager) -> dict[str, KPICollection]:
        """
        Get related KPIs grouped by relationship type (FIXED VERSION).

        Finds KPIs related to the given KPI across different dimensions:
        - Same object/flag, different aggregations
        - Same object/flag/aggregation, different datasets
        - Same object/flag/aggregation, different comparisons/operations

        This is a corrected version that properly separates different relationship types.

        Args:
            kpi: Source KPI to find relatives for
            study_manager: StudyManager for accessing merged KPI collection

        Returns:
            Dict mapping relationship type to KPICollection of related KPIs
        """
        groups = {
            'Different Comparisons / Operations': KPICollection(),
            'Different Aggregations': KPICollection(),
            'Different Datasets': KPICollection(),
        }

        if not study_manager:
            return groups

        # Get reference KPI attributes
        object_name = kpi.attributes.object_name
        flag = kpi.attributes.flag
        model_flag = kpi.attributes.model_flag
        aggregation = kpi.attributes.aggregation
        dataset_name = kpi.attributes.dataset_name
        value_comparison = kpi.attributes.value_comparison
        arithmetic_operation = kpi.attributes.arithmetic_operation

        # Must have flag and aggregation
        if not flag or aggregation is None:
            return groups

        try:
            # Get all KPIs for same object/flag/model_flag
            all_related = study_manager.scen_comp.get_merged_kpi_collection()
            pre_filtered = all_related.filter(
                object_name=object_name,
                flag=flag,
                model_flag=model_flag
            )
        except:
            return groups

        # Determine if main KPI has comparison/operation
        main_has_comparison = value_comparison is not None or arithmetic_operation is not None

        for potential in pre_filtered:
            # Skip self
            if potential is kpi:
                continue

            pot_agg = potential.attributes.aggregation
            pot_dataset = potential.attributes.dataset_name
            pot_comparison = potential.attributes.value_comparison
            pot_operation = potential.attributes.arithmetic_operation
            pot_has_comparison = pot_comparison is not None or pot_operation is not None

            # Category 1: Different Aggregations
            # Same dataset, same comparison status, different aggregation
            if (pot_dataset == dataset_name and
                pot_agg != aggregation and
                pot_comparison == value_comparison and
                pot_operation == arithmetic_operation):
                groups['Different Aggregations'].add(potential)
                continue

            # Category 2: Different Datasets
            # Same aggregation, same comparison status, different dataset
            if (pot_agg == aggregation and
                pot_dataset != dataset_name and
                pot_comparison == value_comparison and
                pot_operation == arithmetic_operation):
                groups['Different Datasets'].add(potential)
                continue

            # Category 3: Different Comparisons/Operations
            # Same dataset, same aggregation, different comparison/operation
            if (pot_dataset == dataset_name and
                pot_agg == aggregation and
                (pot_comparison != value_comparison or pot_operation != arithmetic_operation)):
                groups['Different Comparisons / Operations'].add(potential)
                continue

        return groups


class KPICollectionMapVisualizer:
    """
    High-level KPI collection map visualizer for energy system analysis.

    Main orchestrator for converting KPI collections into organized folium map
    visualizations. Handles KPI grouping, feature group creation, tooltip
    enhancement, and progress tracking.

    Supports multiple generators for complex visualizations (e.g., areas with
    text overlays, lines with arrow indicators) and provides sophisticated
    KPI organization and related KPI discovery for enhanced user experience.

    Args:
        generators: Single generator or list of generators for visualization
        study_manager: StudyManager for enhanced KPI relationships and tooltips
        include_related_kpis_in_tooltip: Add related KPIs to tooltip display
        kpi_grouping_manager: Custom grouping manager (optional)
        **kwargs: Additional arguments passed to data items

    Examples:

        Basic area visualization:
        >>> visualizer = KPICollectionMapVisualizer(
        ...     generators=[
        ...         AreaGenerator(
        ...             AreaFeatureResolver(
        ...                 fill_color=PropertyMapper.from_kpi_value(color_scale),
        ...                 tooltip=True
        ...             )
        ...         )
        ...     ],
        ...     study_manager=study
        ... )
        >>> feature_groups = visualizer.get_feature_groups(price_kpis)
        >>> for fg in feature_groups:
        ...     fg.add_to(map)

        Complex multi-layer visualization:
        >>> visualizer = KPICollectionMapVisualizer(
        ...     generators=[
        ...         AreaGenerator(AreaFeatureResolver(...)),
        ...         TextOverlayGenerator(TextOverlayFeatureResolver(...))
        ...     ],
        ...     include_related_kpis_in_tooltip=True,
        ...     study_manager=study
        ... )
        >>> visualizer.generate_and_add_feature_groups_to_map(
        ...     kpi_collection, folium_map, show='first'
        ... )
    """

    def __init__(
            self,
            generators: FoliumObjectGenerator | List[FoliumObjectGenerator],
            study_manager: 'StudyManager' = None,
            include_related_kpis_in_tooltip: bool = False,
            kpi_grouping_manager: KPIGroupingManager = None,
            value_formatting: VALUE_FORMATTING_OPTIONS = 'per_feature_group',
            **kwargs
    ):
        self.generators: List[FoliumObjectGenerator] = generators if isinstance(generators, list) else [generators]
        self.study_manager = study_manager
        self.include_related_kpis_in_tooltip = include_related_kpis_in_tooltip

        self.grouping_manager = kpi_grouping_manager or KPIGroupingManager()
        self.converter_resolver = KPIValueConverterResolver(value_formatting)

        # Replace tooltip mapper with enhanced version if needed
        if self.include_related_kpis_in_tooltip:
            enhanced_tooltip = self._create_enhanced_tooltip_generator()
            for g in self.generators:
                g.feature_resolver.property_mappers['tooltip'] = enhanced_tooltip

        self.kwargs = kwargs

    def generate_and_add_feature_groups_to_map(
            self,
            kpi_collection: KPICollection,
            folium_map: folium.Map,
            show: SHOW_OPTIONS = 'none',
            overlay: bool = False,
    ) -> list[folium.FeatureGroup]:
        """Generate feature groups and add them to map."""
        fgs = self.get_feature_groups(kpi_collection, show=show, overlay=overlay)
        for fg in fgs:
            folium_map.add_child(fg)
        return fgs

    def get_feature_groups(
            self,
            kpi_collection: KPICollection,
            show: SHOW_OPTIONS = 'none',
            overlay: bool = False
    ) -> list[folium.FeatureGroup]:
        """
        Create feature groups for KPI collection with organized grouping.

        Main method that processes KPI collection through grouping, creates
        folium FeatureGroups, and applies all configured generators to create
        a complete map visualization.

        Args:
            kpi_collection: Collection of KPIs to visualize
            show: Which feature groups to show initially ('first', 'last', 'none')
            overlay: Whether feature groups should be overlay controls

        Returns:
            List of folium FeatureGroup objects ready to add to map
        """
        from tqdm import tqdm
        from mesqual.utils.logging import get_logger

        logger = get_logger(__name__)
        feature_groups = []
        failed = []

        pbar = tqdm(kpi_collection, total=kpi_collection.size, desc=f'{self.__class__.__name__}')
        with pbar:
            kpi_groups = self.grouping_manager.get_kpi_groups(kpi_collection)

            for kpi_group in kpi_groups:
                group_name = self.grouping_manager.get_feature_group_name(kpi_group)

                if show == 'first':
                    show_fg = kpi_group == kpi_groups[0]
                elif show == 'last':
                    show_fg = kpi_group == kpi_groups[-1]
                else:
                    show_fg = False

                fg = folium.FeatureGroup(name=group_name, overlay=overlay, show=show_fg)

                for kpi in kpi_group:
                    # Get converter for this KPI using the resolver
                    converter = self.converter_resolver.get_converter_for_kpi(
                        kpi, group_name, kpi_group
                    )

                    data_item = KPIDataItem(kpi, kpi_collection, study_manager=self.study_manager, **self.kwargs)

                    for generator in self.generators:
                        try:
                            generator.generate(data_item, fg, value_converter=converter)
                        except Exception as e:
                            failed.append((kpi.name, group_name, e))

                    pbar.update(1)

                feature_groups.append(fg)

        if failed:
            logger.warning(
                f'Exception while trying to add {len(failed)} KPIs: {failed[:3]}'
            )
        return feature_groups

    def _create_enhanced_tooltip_generator(self) -> PropertyMapper:
        """
        Create tooltip generator that includes related KPIs.

        Returns a PropertyMapper that accepts value_converter as a parameter,
        enabling dynamic converter selection per KPI.
        """

        def generate_tooltip(data_item: KPIDataItem, value_converter=None) -> str:
            kpi = data_item.kpi
            kpi_name = kpi.get_kpi_name_with_dataset_name()

            from mesqual.units import Units

            # Use converter if provided, otherwise use default formatting
            if value_converter is not None:
                kpi_text = value_converter.convert(kpi.quantity)
            else:
                kpi_quantity = Units.get_quantity_in_pretty_unit(kpi.quantity)
                kpi_text = Units.get_pretty_text_for_quantity(kpi_quantity, thousands_separator=' ')

            html = '<table style="border-collapse: collapse;">\n'
            html += f'  <tr><td style="padding: 4px 8px;"><strong>{kpi_name}</strong></td>' \
                    f'<td style="text-align: right; padding: 4px 8px;">{kpi_text}</td></tr>\n'

            if self.include_related_kpis_in_tooltip and self.study_manager:
                related_groups = self.grouping_manager.get_related_kpi_groups(
                    kpi, self.study_manager
                )

                if any(not g.empty for g in related_groups.values()):
                    for name, group in related_groups.items():
                        if group.empty:
                            continue
                        html += "<tr><p>&nbsp;</p></tr>"
                        html += f'  <tr><th colspan="2" style="text-align: left; padding: 8px;">{name}</th></tr>\n'
                        for related_kpi in group:
                            related_kpi_name = related_kpi.get_kpi_name_with_dataset_name()

                            # Use converter for related KPIs too if provided
                            if value_converter is not None:
                                related_kpi_value_text = value_converter.convert(related_kpi.quantity)
                            else:
                                related_kpi_quantity = Units.get_quantity_in_pretty_unit(related_kpi.quantity)
                                related_kpi_value_text = Units.get_pretty_text_for_quantity(
                                    related_kpi_quantity,
                                    thousands_separator=' ',
                                )
                            html += f'  <tr><td style="padding: 4px 8px;">{related_kpi_name}</td>' \
                                    f'<td style="text-align: right; padding: 4px 8px;">{related_kpi_value_text}</td></tr>\n'

            html += '<br><p>&nbsp;</p></table>'
            return html

        return PropertyMapper(generate_tooltip)
