from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from mesqual.flag import FlagTypeProtocol
    from mesqual.datasets.dataset import Dataset
    from mesqual.kpis.kpi import KPI


class KPIDefinition(ABC):
    """
    Abstract base for KPI specifications.

    Defines WHAT to compute, not HOW to compute efficiently.
    Subclasses implement generate_kpis() to create KPI instances.

    A KPIDefinition is a lightweight specification that can be created
    once and used to generate KPIs for multiple datasets.
    """

    @abstractmethod
    def generate_kpis(self, dataset: Dataset) -> list[KPI]:
        """
        Generate KPI instances from this definition.

        This method is responsible for:
        1. Fetching required data from dataset
        2. Computing values
        3. Creating KPI instances with proper attributes

        Args:
            dataset: Dataset to compute KPIs for

        Returns:
            List of computed KPI instances
        """
        pass

    @abstractmethod
    def required_flags(self) -> set[FlagTypeProtocol]:
        """
        Return set of flags needed for computation.

        Returns:
            Set of flags required by this definition
        """
        pass
