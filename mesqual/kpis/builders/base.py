from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypeVar, Generic

from mesqual.typevars import KPIDefinitionType


class KPIBuilder(ABC, Generic[KPIDefinitionType]):
    """
    Abstract base class for all KPI builders.

    Provides common functionality for fluent builder patterns used in KPI definition creation.
    All builders support optional name prefixes and suffixes for customizing KPI names.

    Subclasses must implement:
    - build(): Generate list of KPI definitions from builder configuration

    Common pattern:

        >>> builder = SomeKPIBuilder()
        >>> definitions = (
        ...     builder
        ...     .some_config_method(...)
        ...     .with_name_prefix("Custom: ")
        ...     .with_name_suffix(" (v2)")
        ...     .with_extra_attributes(my_category='my_cat_1', my_group='my_grp_1')
        ...     .build()
        ... )
    """

    def __init__(self):
        """Initialize builder with common attributes."""
        self._name_prefix: str = ''
        self._name_suffix: str = ''
        self._extra_attributes: dict = dict()
        self._custom_name: str | None = None

    def with_name_prefix(self, prefix: str) -> KPIBuilder:
        """
        Set name prefix for generated KPIs.

        Args:
            prefix: Prefix string to prepend to KPI names

        Returns:
            Self for method chaining
        """
        self._name_prefix = prefix
        return self

    def with_name_suffix(self, suffix: str) -> KPIBuilder:
        """
        Set name suffix for generated KPIs.

        Args:
            suffix: Suffix string to append to KPI names

        Returns:
            Self for method chaining
        """
        self._name_suffix = suffix
        return self

    def with_custom_name(self, name: str) -> KPIBuilder:
        """
        Set custom name that completely overrides automatic name generation.

        When set, the KPI will use this exact name instead of generating one
        from flag, aggregation, object, etc.

        Args:
            name: Custom name for the KPI

        Returns:
            Self for method chaining
        """
        self._custom_name = name
        return self

    def with_extra_attributes(self, **kwargs) -> KPIBuilder:
        """
        Set additional extra (custom) attributes.

        When set, the KPI will contain these attributes under "kpi.attributes.extra_attributes".

        Kwargs:
            kwargs: Custom key-value attributes.

        Returns:
            Self for method chaining
        """
        self._extra_attributes.update(**kwargs)
        return self

    @abstractmethod
    def build(self) -> list[KPIDefinitionType]:
        """
        Generate all KPI definitions from builder configuration.

        Returns:
            List of KPI definition instances
        """
        pass
