import os

from mesqual.datasets.dataset_collection import DatasetConcatCollection, Dataset
from mesqual.datasets.dataset_comparison import DatasetComparison, DatasetConcatCollectionOfComparisons


class StudyManager:
    """
    Central orchestrator for multi-scenario energy systems analysis using MESQUAL's three-tier collection system.
    
    StudyManager provides unified access to scenarios, scenario comparisons, and combined datasets,
    implementing MESQUAL's core architectural principle that "Everything is a Dataset". It manages
    the complete lifecycle of multi-scenario studies including data organization, automatic delta
    computation, and export functionality.
    
    The class implements three primary data access patterns:
        - `.scen`: Access to individual scenario data (DatasetConcatCollection)
        - `.comp`: Access to scenario comparison deltas (DatasetConcatCollectionOfComparisons)
        - `.scen_comp`: Unified access to both scenarios and comparisons with type distinction
    
    Attributes:
        scen (DatasetConcatCollection): Collection of scenario datasets with consistent .fetch() interface
        comp (DatasetConcatCollectionOfComparisons): Collection of scenario comparisons with automatic delta calculation
        scen_comp (DatasetConcatCollection): Unified collection combining scenarios and comparisons
    
    Examples:
        Basic multi-scenario study setup:
        
        >>> from mesqual import StudyManager
        >>> from mesqual_pypsa import PyPSADataset
        >>> 
        >>> study = StudyManager.factory_from_scenarios(
        ...     scenarios=[
        ...         PyPSADataset(base_network, name='base'),
        ...         PyPSADataset(high_res_network, name='high_res'),
        ...         PyPSADataset(low_gas_network, name='cheap_gas'),
        ...     ],
        ...     comparisons=[('high_res', 'base'), ('cheap_gas', 'base')],  # StudyManager handles automatic naming with '*variation_dataset_name* vs *reference_dataset_name*'
        ... )


        Three-tier data access:
        
        >>> # Access scenario data across all scenarios
        >>> scenario_prices = study.scen.fetch('buses_t.marginal_price')
        >>> print(scenario_prices)  # Print price time-series df for all scenarios concatenated with MultiIndex
            dataset              base        ... n_cheap_gas
            Bus                     1    10  ...          99 99_220kV
            snapshot                         ...
            2011-01-01 00:00:00 -0.44  5.77  ...       23.83    23.79
            2011-01-01 01:00:00 -0.58  6.10  ...       22.38    22.33
            ...                   ...   ...  ...         ...      ...
            2011-01-01 22:00:00 20.98 20.44  ...       22.72    22.67
            2011-01-01 23:00:00 13.43 18.94  ...       22.59    22.54

        >>> # Access comparison deltas automatically calculated across all comparisons
        >>> price_changes = study.comp.fetch('buses_t.marginal_price')
        >>> print(price_changes)  # Print price time-series df for all comparisons concatenated with MultiIndex
            dataset             high_res vs base        ... n_cheap_gas vs base
            Bus                                1    10  ...                  99 99_220kV
            snapshot                                    ...
            2011-01-01 00:00:00             0.21  0.02  ...                0.10     0.11
            2011-01-01 01:00:00             0.02  0.06  ...               -0.81    -0.82
            ...                              ...   ...  ...                 ...      ...
            2011-01-01 22:00:00             0.04 -0.05  ...               -2.10    -2.14
            2011-01-01 23:00:00             2.33  0.46  ...               -1.74    -1.78

        >>> # Access unified view with type-level distinction
        >>> unified_scen_and_comp_price_data = study.scen_comp.fetch('buses_t.marginal_price')
        >>> print(unified_scen_and_comp_price_data)  # Print price time-series df for all scenarios and comparisons concatenated with additional level on MultiIndex:
            type                scenario        ...          comparison
            dataset                 base        ... n_cheap_gas vs base
            Bus                        1    10  ...                  99 99_220kV
            snapshot                            ...
            2011-01-01 00:00:00    -0.44  5.77  ...                0.10     0.11
            2011-01-01 01:00:00    -0.58  6.10  ...               -0.81    -0.82
            ..                       ...   ...  ...                 ...      ...
            2011-01-01 22:00:00    20.98 20.44  ...               -2.10    -2.14
            2011-01-01 23:00:00    13.43 18.94  ...               -1.74    -1.78


        Access individual datasets:

        >>> # Access to data from individual scenario
        >>> ds_base = study.scen.get_dataset('base')
        >>> base_prices = ds_base.fetch('buses_t.marginal_price')
        >>> print(base_prices)  # Print price time-series df for base scenario
            Bus                     1    10  ...    99  99_220kV
            snapshot                         ...
            2011-01-01 00:00:00 -0.44  5.77  ... 23.72     23.69
            2011-01-01 01:00:00 -0.58  6.10  ... 23.19     23.14
            ...                   ...   ...  ...   ...       ...
            2011-01-01 22:00:00 20.98 20.44  ... 24.81     24.81
            2011-01-01 23:00:00 13.43 18.94  ... 24.33     24.32

        >>> base_buses_model_df = ds_base.fetch('buses')
        >>> print(base_buses_model_df)  # Print bus model df for base scenario
                       v_nom type  ...  v_mag_pu_max  control
            Bus                    ...
            1         220.00       ...           inf    Slack
            2         380.00       ...           inf       PQ
            ...          ...  ...  ...           ...      ...
            450_220kV 220.00       ...           inf       PQ
            458_220kV 220.00       ...           inf       PQ

        >>> # Access to individual comparison dataset
        >>> ds_comp_high_res = study.comp.get_dataset('high_res vs base')
        >>> price_changes_high_res = ds_comp_high_res.fetch('buses_t.marginal_price')
        >>> print(price_changes_high_res)  # Print price time-series df changes (deltas) comparing high_res to base
            Bus                    1    10  ...    99  99_220kV
            snapshot                        ...
            2011-01-01 00:00:00 0.21  0.02  ...  0.01      0.01
            2011-01-01 01:00:00 0.02  0.06  ... -0.07     -0.07
            ...                  ...   ...  ...   ...       ...
            2011-01-01 22:00:00 0.04 -0.05  ... -0.00     -0.00
            2011-01-01 23:00:00 2.33  0.46  ... -0.09     -0.09

        >>> bus_model_changes_high_res = ds_comp_high_res.fetch('buses')
        >>> print(bus_model_changes_high_res)  # Print bus model df changes comparing high_res to base
                       v_nom type  ...  v_mag_pu_max  control
            Bus                    ...
            1           0.00       ...           NaN    Slack
            2           0.00       ...           NaN       PQ
            ...          ...  ...  ...           ...      ...
            450_220kV   0.00       ...           NaN       PQ
            458_220kV   0.00       ...           NaN       PQ


        Access to merged (model) dfs for all sub datasets:

        >>> # Access merged dataframe (useful to get unified model_df in case of different objects across scenarios)
        >>> bus_model_df_merged = study.scen.fetch_merged('buses')
        >>> print(bus_model_df_merged)  # Print merged bus model df for all scenarios
                       v_nom type  ...  v_mag_pu_max  control
            Bus                    ...
            1         220.00       ...           inf    Slack
            2         380.00       ...           inf       PQ
            ...          ...  ...  ...           ...      ...
            450_220kV 220.00       ...           inf       PQ
            458_220kV 220.00       ...           inf       PQ

    Notes:
        - All collections share the same .fetch(flag) interface for consistent data access
        - Scenario comparisons are computed automatically as deltas (variation - reference)
        - The unified collection uses 'type' as the concat_level_name to distinguish data sources
        - Supports dynamic addition of scenarios and comparisons after initialization
    """
    def __init__(
            self,
            scenarios: DatasetConcatCollection,
            comparisons: DatasetConcatCollectionOfComparisons,
            export_folder: str = None,
    ):
        self._scenarios = scenarios
        self._comparisons = comparisons
        self._scenarios_and_comparisons: DatasetConcatCollection = DatasetConcatCollection(
            [
                scenarios,
                comparisons
            ],
            name='scenarios_and_comparisons',
            concat_level_name='type',
        )
        self._export_folder: str = export_folder
        if export_folder is not None:
            self._ensure_folder_exists(export_folder)

    @property
    def scen(self) -> DatasetConcatCollection:
        return self._scenarios

    @property
    def scenario_names(self) -> list[str]:
        return [s.name for s in self.scen.dataset_iterator]

    def add_scenario(self, dataset: Dataset):
        self._scenarios.add_dataset(dataset)

    @property
    def comp(self) -> DatasetConcatCollectionOfComparisons:
        return self._comparisons

    @property
    def comparison_names(self) -> list[str]:
        return [c.name for c in self.comp.dataset_iterator]

    def add_comparison(self, dataset: DatasetComparison):
        self._comparisons.add_dataset(dataset)

    @property
    def scen_comp(self) -> DatasetConcatCollection:
        return self._scenarios_and_comparisons

    @property
    def export_folder(self) -> str:
        return self._export_folder

    @export_folder.setter
    def export_folder(self, folder_path: str):
        self._ensure_folder_exists(folder_path)
        self._export_folder = folder_path

    @staticmethod
    def _ensure_folder_exists(folder: str):
        os.makedirs(folder, exist_ok=True)

    def export_path(self, file_name: str) -> str:
        if self._export_folder is None:
            raise RuntimeError(f'Export folder must be assigned first.')
        return os.path.join(self._export_folder, file_name)

    @classmethod
    def factory_from_scenarios(
            cls,
            scenarios: list[Dataset],
            comparisons: list[tuple[str, str]],
            export_folder: str = None
    ) -> 'StudyManager':
        scen = DatasetConcatCollection(scenarios, name='scenario', concat_level_name='dataset',)
        comp = DatasetConcatCollectionOfComparisons(
            datasets=[
                DatasetComparison(scen.get_dataset(var), scen.get_dataset(ref))
                for var, ref in comparisons
            ],
            name='comparison',
            concat_level_name='dataset',
        )
        return cls(scen, comp, export_folder)
