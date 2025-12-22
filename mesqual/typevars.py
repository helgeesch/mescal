from __future__ import annotations
from typing import TypeVar, TYPE_CHECKING


if TYPE_CHECKING:
    pass

FlagType = TypeVar('FlagType', bound='FlagTypeProtocol')
DatasetType = TypeVar('DatasetType', bound='Dataset')
DatasetConfigType = TypeVar('DatasetConfigType', bound='DatasetConfig')
FlagIndexType = TypeVar('FlagIndexType', bound='FlagIndex')
KPIType = TypeVar('KPIType', bound='KPI')
KPIDefinitionType = TypeVar('KPIDefinitionType', bound='KPIDefinition')
ValueOperationType = TypeVar('ValueOperationType', bound='OperationOfTwoValues')
FeatureResolverType = TypeVar('FeatureResolverType', bound='FeatureResolver')
ResolvedFeatureType = TypeVar('ResolvedFeatureType', bound='ResolvedFeature')
ValueMappingType = TypeVar('ValueMappingType', bound='BaseMapping')
DiscreteMappingType = TypeVar('DiscreteMappingType', bound='DiscreteInputMapping')
ContinuousMappingType = TypeVar('ContinuousMappingType', bound='SegmentedContinuousInputMappingBase')
FoliumLegendType = TypeVar('FoliumLegendType', bound='BaseLegend')
