from typing import Dict, List, Tuple, Optional, Union

import numpy as np
from branca.element import MacroElement, Template


class SegmentedOpacityMap:
    """PLACEHOLDER"""  # TODO: implement sophisticated version

    """Maps values to opacity using customizable segments similar to SegmentedColorMap."""

    def __init__(
            self,
            value: float
    ):
        if not value <= 1 and value >= 0:
            raise ValueError('Value must be between 0 and 1.')
        self.value = value

    def __call__(self, value: float) -> float:
        """Get the appropriate width for a value."""
        return self.value


class SegmentedOpacityMapLegend(SegmentedOpacityMap, MacroElement):
    pass
