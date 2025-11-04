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
        self.value_formatting = value_formatting

        self.grouping_manager = kpi_grouping_manager or KPIGroupingManager()

        # Enhanced tooltip if needed
        if self.include_related_kpis_in_tooltip:
            for g in self.generators:
                g.tooltip_generator = self._create_enhanced_tooltip_generator()
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

    def _create_converter_for_feature_group(self, kpi_group: KPICollection) -> QuantityToTextConverter:
        """Create converter configured for a specific feature group."""
        quantities = [kpi.quantity for kpi in kpi_group]
        return QuantityToTextConverter.from_quantities(
            quantities,
            thousands_separator=' '
        )

    def _create_converter_per_collection(self, kpi_collection: KPICollection) -> dict[str, QuantityToTextConverter]:
        """Create converters per base unit for the entire collection."""
        from mesqual.units import Units

        # Group KPIs by base unit
        kpis_by_base_unit = {}
        for kpi in kpi_collection:
            base_unit = Units.get_base_unit_for_unit(kpi.quantity.units)
            base_unit_str = str(base_unit)
            if base_unit_str not in kpis_by_base_unit:
                kpis_by_base_unit[base_unit_str] = []
            kpis_by_base_unit[base_unit_str].append(kpi)

        # Create converter for each base unit group
        converters = {}
        for base_unit_str, kpis in kpis_by_base_unit.items():
            quantities = [kpi.quantity for kpi in kpis]
            converters[base_unit_str] = QuantityToTextConverter.from_quantities(
                quantities,
                thousands_separator=' '
            )

        return converters

    def _get_converter_for_kpi(
        self,
        kpi: KPI,
        kpi_group: KPICollection = None,
        collection_converters: dict[str, QuantityToTextConverter] = None
    ) -> QuantityToTextConverter:
        """Get the appropriate converter for a KPI based on value_formatting strategy."""
        from mesqual.units import Units

        # If explicit converter provided, use it
        if isinstance(self.value_formatting, QuantityToTextConverter):
            return self.value_formatting

        # Per feature group strategy
        if self.value_formatting == 'per_feature_group' and kpi_group is not None:
            return self._create_converter_for_feature_group(kpi_group)

        # Per collection strategy
        if self.value_formatting == 'per_collection' and collection_converters is not None:
            base_unit = Units.get_base_unit_for_unit(kpi.quantity.units)
            base_unit_str = str(base_unit)
            return collection_converters.get(base_unit_str)

        # Fallback: create converter for single KPI
        return QuantityToTextConverter.from_quantities([kpi.quantity], thousands_separator=' ')

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

        # Pre-compute converters for per_collection strategy
        collection_converters = None
        if self.value_formatting == 'per_collection':
            collection_converters = self._create_converter_per_collection(kpi_collection)

        failed = []
        pbar = tqdm(kpi_collection, total=kpi_collection.size, desc=f'{self.__class__.__name__}')
        with pbar:
            kpi_groups = self.grouping_manager.get_kpi_groups(kpi_collection)

            # Cache converter per feature group for efficiency
            group_converters = {}

            for kpi_group in kpi_groups:
                group_name = self.grouping_manager.get_feature_group_name(kpi_group)

                if show == 'first':
                    show_fg = kpi_group == kpi_groups[0]
                elif show == 'last':
                    show_fg = kpi_group == kpi_groups[-1]
                else:
                    show_fg = False

                fg = folium.FeatureGroup(name=group_name, overlay=overlay, show=show_fg)

                # Get/create converter for this group
                if self.value_formatting == 'per_feature_group':
                    if group_name not in group_converters:
                        group_converters[group_name] = self._create_converter_for_feature_group(kpi_group)
                    group_converter = group_converters[group_name]
                else:
                    group_converter = None

                for kpi in kpi_group:
                    # Get appropriate converter for this KPI
                    converter = self._get_converter_for_kpi(
                        kpi,
                        kpi_group=kpi_group,
                        collection_converters=collection_converters
                    )

                    data_item = KPIDataItem(kpi, kpi_collection, study_manager=self.study_manager, **self.kwargs)
                    for generator in self.generators:
                        # Store original value_converter
                        original_converter = generator.feature_resolver.value_converter

                        # Temporarily set converter for this KPI
                        generator.feature_resolver.value_converter = converter

                        if self.include_related_kpis_in_tooltip:
                            _tmp = generator.feature_resolver.property_mappers.get('tooltip', None)
                            generator.feature_resolver.property_mappers['tooltip'] = self._create_enhanced_tooltip_generator(converter)
                        try:
                            generator.generate(data_item, fg)
                        except Exception as e:
                            failed.append((kpi.name, group_name, e))

                        finally:
                            # Restore original converter
                            generator.feature_resolver.value_converter = original_converter

                            if self.include_related_kpis_in_tooltip:
                                if _tmp is not None:
                                    generator.feature_resolver.property_mappers['tooltip'] = _tmp
                                else:
                                    generator.feature_resolver.property_mappers.pop('tooltip', None)
                    pbar.update(1)

                feature_groups.append(fg)
        if failed:
            logger.warning(
                f'Exception while trying to add {len(failed)} KPIs: {failed[:3]}'
            )
        return feature_groups

    def _create_enhanced_tooltip_generator(self, converter: QuantityToTextConverter = None) -> PropertyMapper:
        """Create tooltip generator that includes related KPIs.

        Args:
            converter: Optional QuantityToTextConverter for formatting KPI values
        """

        def generate_tooltip(data_item: KPIDataItem) -> str:

            kpi = data_item.kpi
            kpi_name = kpi.get_kpi_name_with_dataset_name()

            from mesqual.units import Units

            # Use converter if provided, otherwise use default formatting
            if converter is not None:
                kpi_text = converter.convert(kpi.quantity)
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
                            if converter is not None:
                                related_kpi_value_text = converter.convert(related_kpi.quantity)
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
