from abc import ABC, abstractmethod
from typing import List, Tuple, Literal, Union, Any, TYPE_CHECKING

import folium

from mesqual.kpis import KPICollection, KPI
from mesqual.visualizations.folium_viz_system.base_viz_system import FoliumObjectGenerator
from mesqual.visualizations.folium_viz_system.visualizable_data_item import KPIDataItem

if TYPE_CHECKING:
    from mesqual.study_manager import StudyManager
    from mesqual.datasets import DatasetCollection


class CustomKPIGroupGenerator(ABC):
    """
    Abstract base class for custom KPI grouping and visualization routing.

    Provides a unified framework for scenarios where the standard KPIGroupingManager
    and automatic generator application don't fit your needs. This class combines
    two responsibilities in one place:

    1. **Custom grouping logic**: Define which KPIs belong in which feature groups
    2. **Generator routing**: Specify which generators apply to each KPI type

    By putting both concerns in a single class, users have a clear template for
    implementing custom visualization scenarios without juggling multiple abstractions.

    Common Use Cases:
        - **Overlapping groups**: Same KPI appears in multiple feature groups
          (e.g., flow data shown with both prices and net positions)
        - **Type-based visualization**: Different KPI types get different visual styles
          (e.g., areas for prices, arrows for flows)
        - **Custom naming logic**: Feature group names beyond simple attribute composition
        - **Complex grouping criteria**: Multi-dimensional or hierarchical organization

    Template Usage:
        1. Subclass CustomKPIGroupGenerator
        2. Implement create_kpi_groups_with_names() - define your groups
        3. Implement get_generators_for_kpi() - define visualization per KPI type
        4. Call generate_and_add_feature_groups_to_map() to create the map

    Examples:
        Dual groups with overlapping flows:
        >>> class DualGroupGenerator(CustomKPIGroupGenerator):
        ...     def __init__(self, area_gen, arrow_gen, text_gen):
        ...         self.area_gen = area_gen
        ...         self.arrow_gen = arrow_gen
        ...         self.text_gen = text_gen
        ...
        ...     def create_kpi_groups_with_names(self, source):
        ...         # Work with StudyManager directly
        ...         groups = []
        ...         for dataset in source.scen.dataset_iterator:
        ...             kpi_col = dataset.kpi_collection
        ...             price_kpis = kpi_col.filter(flag='price')
        ...             flow_kpis = kpi_col.filter(flag='flow')
        ...
        ...             # Price group: prices + flows
        ...             price_group = price_kpis + flow_kpis
        ...             groups.append((f"{dataset.name} - Prices", price_group))
        ...
        ...             # Net position group: net_positions + flows
        ...             netpos_kpis = kpi_col.filter(flag='net_position')
        ...             netpos_group = netpos_kpis + flow_kpis
        ...             groups.append((f"{dataset.name} - Net Positions", netpos_group))
        ...         return groups
        ...
        ...     def get_generators_for_kpi(self, kpi):
        ...         if 'price' in kpi.attributes.flag:
        ...             return [self.area_gen, self.text_gen]
        ...         elif 'net_position' in kpi.attributes.flag:
        ...             return [self.area_gen, self.text_gen]
        ...         elif 'flow' in kpi.attributes.flag:
        ...             return [self.arrow_gen]
        ...         return []
        ...
        >>> generator = DualGroupGenerator(area_gen, arrow_gen, text_gen)
        >>> generator.generate_and_add_feature_groups_to_map(
        ...     study_manager, map_obj
        ... )

        Working with merged KPI collection:
        >>> class RegionalGenerator(CustomKPIGroupGenerator):
        ...     def create_kpi_groups_with_names(self, source):
        ...         # Work with KPICollection directly
        ...         groups = []
        ...         for region in ['North', 'South', 'East', 'West']:
        ...             regional_kpis = source.filter(region=region)
        ...             groups.append((f"Region: {region}", regional_kpis))
        ...         return groups
        ...
        ...     def get_generators_for_kpi(self, kpi):
        ...         return [self.circle_marker_gen]
    """

    @abstractmethod
    def create_kpi_groups_with_names(
        self,
        source: Union['KPICollection', 'StudyManager', 'DatasetCollection', Any]
    ) -> List[Tuple[str, KPICollection]]:
        """
        Define custom grouping logic and feature group names.

        This is where you implement your specific grouping strategy. The method
        receives a flexible source (KPICollection, StudyManager, DatasetCollection,
        or any custom object) and returns a list of (name, kpi_group) tuples that
        define what appears in each feature group.

        Args:
            source: Flexible source for KPI data. Can be:
                - KPICollection: Merged collection to filter
                - StudyManager: Access study.scen/study.comp datasets directly
                - DatasetCollection: Iterate through datasets (e.g., study.scen)
                - Any: Custom source you define

        Returns:
            List of (group_name, kpi_group) tuples where:
            - group_name: String to display in the map layer control
            - kpi_group: KPICollection containing KPIs for this feature group

        Notes:
            - KPIs can appear in multiple groups (they will be visualized multiple times)
            - Return empty list if no valid groups can be created
            - Group order in the list determines display order in layer control
            - Group names should be descriptive and unique

        Examples:
            Using StudyManager to iterate datasets:
            >>> def create_kpi_groups_with_names(self, source: StudyManager):
            ...     groups = []
            ...     for dataset in source.scen.dataset_iterator:
            ...         kpi_col = dataset.kpi_collection  # dataset.get_merged_kpi_collection() if DatasetCollection
            ...         price_kpis = kpi_col.filter(flag='price')
            ...         flow_kpis = kpi_col.filter(flag='flow')
            ...         combined = price_kpis + flow_kpis
            ...         groups.append((f"{dataset.name} - Combined", combined))
            ...     return groups

            Using merged KPICollection:
            >>> def create_kpi_groups_with_names(self, source: KPICollection):
            ...     groups = []
            ...     datasets = source.get_all_kpi_attributes_and_value_sets()['dataset_name']
            ...     for dataset in datasets:
            ...         kpis = source.filter(dataset_name=dataset, flag='price')
            ...         groups.append((f"{dataset} - Prices", kpis))
            ...     return groups

            Using DatasetCollection:
            >>> def create_kpi_groups_with_names(self, source: DatasetCollection):
            ...     groups = []
            ...     for dataset in source.dataset_iterator:
            ...         all_kpis = dataset.kpi_collection
            ...         groups.append((f"{dataset.name}", all_kpis))
            ...     return groups
        """
        pass

    @abstractmethod
    def get_generators_for_kpi(
        self,
        kpi: KPI
    ) -> List[FoliumObjectGenerator]:
        """
        Determine which generators should visualize a specific KPI.

        This method is called for each KPI in each group, allowing you to route
        different KPI types to different visualization styles. You can return
        multiple generators to create layered visualizations (e.g., area + text).

        Args:
            kpi: The KPI to determine generators for

        Returns:
            List of FoliumObjectGenerator instances to apply to this KPI.
            Return empty list to skip visualization for this KPI.

        Notes:
            - Inspect kpi.attributes to make routing decisions
            - Common attributes: flag, dataset_name, aggregation, object_name
            - Can return empty list to skip certain KPIs
            - Generators are applied in order (useful for layering)

        Example:
            >>> def get_generators_for_kpi(self, kpi):
            ...     # Route based on flag
            ...     if 'price' in kpi.attributes.flag:
            ...         return [self.area_generator, self.text_overlay_generator]
            ...     elif 'flow' in kpi.attributes.flag:
            ...         return [self.arrow_generator]
            ...     elif 'capacity' in kpi.attributes.flag:
            ...         return [self.circle_marker_generator]
            ...     else:
            ...         return []  # Skip unknown types
        """
        pass

    def generate_and_add_feature_groups_to_map(
        self,
        source: Union['KPICollection', 'StudyManager', 'DatasetCollection', Any],
        map_obj: folium.Map,
        show: Literal['first', 'last', 'all', 'none'] = 'first',
        overlay: bool = False,
        **data_item_kwargs
    ) -> List[folium.FeatureGroup]:
        """
        Generate feature groups from source and add them to the map.

        This concrete method orchestrates the entire visualization pipeline:
        1. Calls your create_kpi_groups_with_names() to get groups
        2. For each group, creates a folium FeatureGroup
        3. For each KPI, calls your get_generators_for_kpi() to get generators
        4. Applies generators to create map features
        5. Adds everything to the map

        Args:
            source: Flexible source for KPI data (KPICollection, StudyManager,
                DatasetCollection, or any custom object). Passed to your
                create_kpi_groups_with_names() implementation.
            map_obj: Folium map to add feature groups to
            show: Which feature groups to show initially:
                - 'first': Only first group visible (default)
                - 'last': Only last group visible
                - 'all': All groups visible
                - 'none': No groups visible initially
            overlay: Whether feature groups are overlay controls (default: True)
            **data_item_kwargs: Additional arguments passed to KPIDataItem constructor
                (e.g., study_manager for enhanced tooltips)

        Returns:
            List of folium.FeatureGroup objects that were added to the map

        Examples:

            Passing StudyManager:
            >>> generator: CustomKPIGroupGenerator = MyCustomGroupGenerator()
            >>> feature_groups = generator.generate_and_add_feature_groups_to_map(
            ...     source=study,  # StudyManager instance
            ...     map_obj=m,
            ...     show='first'
            ... )

            Passing merged KPICollection:
            >>> kpi_collection = study.scen.get_merged_kpi_collection()
            >>> generator.generate_and_add_feature_groups_to_map(
            ...     source=kpi_collection,
            ...     map_obj=m,
            ...     show='all'
            ... )

            Passing DatasetCollection:
            >>> generator.generate_and_add_feature_groups_to_map(
            ...     source=study.scen,  # DatasetCollection
            ...     map_obj=m,
            ...     study_manager=study  # For enhanced tooltips
            ... )
        """
        # Get custom groups from subclass
        groups_with_names = self.create_kpi_groups_with_names(source)

        feature_groups = []
        total_groups = len(groups_with_names)

        for idx, (group_name, kpi_group) in enumerate(groups_with_names):
            # Determine if this group should be shown initially
            show_fg = self._should_show_group(idx, total_groups, show)

            # Create feature group
            fg = folium.FeatureGroup(name=group_name, overlay=overlay, show=show_fg)

            # Process each KPI in the group
            for kpi in kpi_group:
                # Get generators for this specific KPI from subclass
                generators = self.get_generators_for_kpi(kpi)

                # Create data item for this KPI
                data_item = KPIDataItem(kpi, kpi_group, **data_item_kwargs)

                # Apply each generator
                for generator in generators:
                    try:
                        generator.generate(data_item, fg)
                    except Exception as e:
                        # Log or handle generation errors without stopping the loop
                        # This matches the behavior in KPICollectionMapVisualizer
                        continue

            feature_groups.append(fg)
            map_obj.add_child(fg)

        return feature_groups

    @staticmethod
    def _should_show_group(idx: int, total: int, show: str) -> bool:
        """
        Determine if a feature group should be shown initially.

        Args:
            idx: Index of the feature group (0-based)
            total: Total number of feature groups
            show: Show strategy ('first', 'last', 'all', 'none')

        Returns:
            True if the group should be visible initially
        """
        if show == 'first':
            return idx == 0
        elif show == 'last':
            return idx == total - 1
        elif show == 'all':
            return True
        else:  # 'none'
            return False