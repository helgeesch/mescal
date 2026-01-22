"""
Dataset module providing the core data access layer for MESQUAL.

This module implements the foundational "Everything is a Dataset" principle,
where all data sources—individual scenarios, merged scenarios, collections,
and comparisons—share a unified interface through the `.fetch(flag)` pattern.

Core Classes:
    Dataset: Abstract base class defining the universal data access interface.
        All dataset types inherit from this class and implement the `fetch()` method.

    DatasetCollection: Base class for grouping related datasets together.
        Provides iteration and batch operations across multiple datasets.

    DatasetLinkCollection: Collection maintaining parent-child relationships.
        Used when datasets need to reference back to their container.

    DatasetMergeCollection: Combines multiple datasets by merging their data.
        Useful for aggregating results from different simulation runs.

    DatasetSumCollection: Aggregates datasets by summing numeric values.
        Commonly used for capacity or production totals across scenarios.

    DatasetConcatCollection: Concatenates datasets along a specified axis.
        Creates MultiIndex structures preserving scenario identities.

    DatasetComparison: Computes differences between scenario pairs.
        Enables delta analysis and comparative studies.

    DatasetConcatCollectionOfComparisons: Specialized collection for comparisons.
        Facilitates systematic comparison across multiple scenario pairs.

    PlatformDataset: Dataset subclass for platform-specific implementations.
        Extended by platform interfaces (e.g., PyPSA, PLEXOS) to provide
        platform-aware data access.

    DatasetConfig: Configuration class controlling dataset behavior.
        Manages caching, post-processing, and platform-specific options.

Example:
    Basic usage pattern::

        from mesqual.datasets import Dataset, DatasetConfig

        # Fetch data from a dataset
        prices = dataset.fetch('buses_t.marginal_price')

        # Configure dataset behavior
        config = DatasetConfig(use_database=True)
        dataset.set_instance_config(config)

        # Work with collections
        for scenario in collection:
            data = scenario.fetch('generators_t.p')

See Also:
    - :mod:`mesqual.flag`: Flag types and flag index implementations
    - :mod:`mesqual.kpis`: KPI calculation framework
    - :mod:`mesqual.databases`: Caching backends for datasets
"""
from mesqual.datasets.dataset import Dataset
from mesqual.datasets.dataset_collection import (
    DatasetCollection,
    DatasetLinkCollection,
    DatasetMergeCollection,
    DatasetSumCollection,
    DatasetConcatCollection,
)
from mesqual.datasets.dataset_comparison import DatasetComparison, DatasetConcatCollectionOfComparisons
from mesqual.datasets.platform_dataset import PlatformDataset
from mesqual.datasets.dataset_config import DatasetConfig


__all__ = [
    'Dataset',
    'DatasetCollection',
    'DatasetLinkCollection',
    'DatasetMergeCollection',
    'DatasetSumCollection',
    'DatasetConcatCollection',
    'DatasetComparison',
    'DatasetConcatCollectionOfComparisons',
    'PlatformDataset',
    'DatasetConfig',
]
