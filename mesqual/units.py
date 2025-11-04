from typing import Iterator, ClassVar
import numpy as np
from pint import get_application_registry, UnitRegistry, set_application_registry, Unit, Quantity

from mesqual.enums import QuantityTypeEnum


class UnitNotFound(Exception):
    pass


class UnitRegistryNotComplete(Exception):
    def __init__(self, message: str = None):
        base = f'You should never end up here. Your units are not properly registered in the {Units.__name__} class.'
        message = message or ''
        super().__init__(base + message)


ureg = UnitRegistry(on_redefinition='ignore')

ureg.define("Wh = [energy]")
ureg.define("kWh = 1e3 Wh = kWh")
ureg.define("MWh = 1e6 Wh = MWh")
ureg.define("GWh = 1e9 Wh = GWh")
ureg.define("TWh = 1e12 Wh = TWh")

ureg.define("W = [power]")
ureg.define("kW = 1e3 W = kW")
ureg.define("MW = 1e6 W = MW")
ureg.define("GW = 1e9 W = GW")
ureg.define("TW = 1e12 W = TW")

ureg.define("EUR = [currency]")
ureg.define("EUR_cent = 1e-2 EUR")
ureg.define("kEUR = 1e3 EUR = kEUR")
ureg.define("MEUR = 1e6 EUR = MEUR")
ureg.define("BEUR = 1e9 EUR = BEUR")
ureg.define("TEUR = 1e12 EUR = TEUR")

ureg.define("EUR_per_W = EUR / W = [price_for_capacity]")
ureg.define("EUR_per_MW = EUR / MW")

ureg.define("EUR_per_Wh = EUR / Wh = [price_for_energy]")
ureg.define("EUR_per_MWh = EUR / MWh")

ureg.define("minute = [time]")
ureg.define("hour = 60 minute = hour")
ureg.define("day = 24 hour = day")
ureg.define("week = 7 day = week")
ureg.define("year = 365 day = year")

ureg.define("W_per_min = W / minute = [ramping]")
ureg.define("MW_per_min = MW / minute")
ureg.define("MW_per_hour = MW / hour")

ureg.define("EUR_per_W_per_min = EUR / (W / minute) = [price_for_ramping]")
ureg.define("EUR_per_MW_per_min = EUR / (MW / minute)")
ureg.define("EUR_per_MW_per_hour = EUR / (MW / hour)")

ureg.define("MTU = [mtu]")
ureg.define("per_unit = [pu]")
ureg.define("perc = [percentage]")
ureg.define("percent_base = 1e-2 percent = percent_base")


ureg.define("NaU = []")  # Not a Unit; no physical meaning, dimensionless
ureg.define("MissingUnit = []")  # For missing units


class _IterableUnitsMeta(type):
    def __iter__(cls) -> Iterator[Unit]:
        return (u for name, u in cls.__dict__.items() if isinstance(u, Unit))


class Units(metaclass=_IterableUnitsMeta):
    _ureg = ureg
    Unit = Unit
    Quantity = Quantity

    Wh = _ureg.Wh
    kWh = _ureg.kWh
    MWh = _ureg.MWh
    GWh = _ureg.GWh
    TWh = _ureg.TWh

    W = _ureg.W
    kW = _ureg.kW
    MW = _ureg.MW
    GW = _ureg.GW
    TW = _ureg.TW

    W_per_min = _ureg.W_per_min
    MW_per_min = _ureg.MW_per_min
    MW_per_hour = _ureg.MW_per_hour

    EUR = _ureg.EUR
    kEUR = _ureg.kEUR
    MEUR = _ureg.MEUR
    BEUR = _ureg.BEUR
    TEUR = _ureg.TEUR

    EUR_per_W = _ureg.EUR_per_W
    EUR_per_MW = _ureg.EUR_per_MW

    EUR_per_Wh = _ureg.EUR_per_Wh
    EUR_per_MWh = _ureg.EUR_per_MWh

    EUR_per_W_per_min = _ureg.EUR_per_W_per_min
    EUR_per_MW_per_min = _ureg.EUR_per_MW_per_min
    EUR_per_MW_per_hour = _ureg.EUR_per_MW_per_hour

    percent_base = _ureg.percent_base
    percent = _ureg.perc
    per_unit = _ureg.per_unit
    MTU = _ureg.MTU
    NaU = _ureg.NaU
    MissingUnit = _ureg.MissingUnit

    _STRING_REPLACEMENTS = {
        '_per_': '/',
        'EUR': '€',
        'per_unit': 'pu',
        'perc': '%',
        'inf': '∞',
        'nan': 'N/A',
    }

    _INTENSIVE_QUANTITIES = [W, EUR_per_Wh, percent_base, per_unit]
    _EXTENSIVE_QUANTITIES = [Wh, EUR, MTU]

    @classmethod
    def get_quantity_type_enum(cls, unit: Unit) -> QuantityTypeEnum:
        base_unit = cls.get_base_unit_for_unit(unit)
        if base_unit in cls._INTENSIVE_QUANTITIES:
            return QuantityTypeEnum.INTENSIVE
        elif base_unit in cls._EXTENSIVE_QUANTITIES:
            return QuantityTypeEnum.EXTENSIVE
        raise KeyError(f'QuantityTypeEnum for {unit} not registered')

    @classmethod
    def units_have_same_base(cls, unit_1: Unit, unit_2: Unit) -> bool:
        return unit_1.dimensionality == unit_2.dimensionality

    @classmethod
    def get_base_unit_for_unit(cls, unit: Unit) -> Unit:
        return cls.get_target_unit_for_oom(unit, 1)

    @classmethod
    def get_oom_of_unit(cls, unit: Unit) -> float:
        return (1 * unit).to_base_units().magnitude

    @classmethod
    def get_target_unit_for_oom(cls, reference_unit: Unit, target_oom: float) -> Quantity:
        units = cls.get_all_units_with_equal_base(reference_unit)
        for u in units:
            if cls.get_oom_of_unit(u) == target_oom:
                return u
        raise UnitNotFound(f'No unit with order of mag {target_oom:.0e} for {reference_unit}')

    @classmethod
    def get_closest_unit_for_oom(cls, reference_unit: Unit, target_oom: float) -> Quantity:
        units_with_same_dimension = cls.get_all_units_with_equal_base(reference_unit)
        if len(units_with_same_dimension) == 0:
            raise UnitRegistryNotComplete
        base_unit = cls.get_base_unit_for_unit(reference_unit)
        sorted_units = sorted(units_with_same_dimension, key=lambda x: (1 * x).to(base_unit).magnitude, reverse=True)
        for u in sorted_units:
            if (1 * u).to(base_unit).magnitude <= target_oom:
                return u
        return sorted_units[0]

    @classmethod
    def get_quantity_in_target_oom(cls, quantity: Quantity, target_oom: float) -> Quantity:
        try:
            target_unit = cls.get_target_unit_for_oom(quantity.units, target_oom)
            return quantity.to(target_unit)
        except UnitNotFound:
            RuntimeWarning(f'# TODO:')
            return quantity

    @classmethod
    def get_quantity_in_target_unit(cls, quantity: Quantity, target_unit: Unit) -> Quantity:
        return quantity.to(target_unit)

    @classmethod
    def get_quantity_in_pretty_unit(cls, quantity: Quantity) -> Quantity:
        base_unit = cls.get_base_unit_for_unit(quantity.units)
        units = cls.get_all_units_with_equal_base(base_unit)
        units = sorted(units, key=lambda x: (1 * x).to(base_unit).magnitude, reverse=False)
        for u in units:
            if abs(quantity.to(u).magnitude) < 10_000:
                return quantity.to(u)
        return quantity.to(units[-1])

    @classmethod
    def get_common_pretty_unit_for_quantities(cls, quantities: list[Quantity]) -> Unit:
        """
        Find common "pretty" unit for a collection of quantities.

        Strategy:
        1. Verify all quantities have same dimensionality
        2. Find median order of magnitude
        3. Select unit closest to median OOM
        4. Ensure most values fit well (< 10,000 in magnitude)

        Args:
            quantities: List of quantities with same dimensionality

        Returns:
            Pretty unit that works well for the collection

        Raises:
            ValueError: If quantities have different dimensionalities or list is empty

        Examples:
            >>> quantities = [1_000_000 * Units.EUR, 5_000_000 * Units.EUR]
            >>> Units.get_common_pretty_unit_for_quantities(quantities)
            Units.MEUR
        """
        if not quantities:
            raise ValueError("Cannot find common unit for empty list of quantities")

        # Verify all quantities have same dimensionality
        base_unit = cls.get_base_unit_for_unit(quantities[0].units)
        for q in quantities[1:]:
            if not cls.units_have_same_base(q.units, base_unit):
                raise ValueError(
                    f"All quantities must have same dimensionality. "
                    f"Found {q.units} which differs from {base_unit}"
                )

        # Convert all to base unit and get magnitudes
        magnitudes = [abs(q.to(base_unit).magnitude) for q in quantities]

        # Filter out zeros for median calculation
        non_zero_magnitudes = [m for m in magnitudes if m > 0]
        if not non_zero_magnitudes:
            # All values are zero, return base unit
            return base_unit

        # Find median order of magnitude
        median_magnitude = np.median(non_zero_magnitudes)

        # Get all available units with same base
        available_units = cls.get_all_units_with_equal_base(base_unit)
        if not available_units:
            return base_unit

        # Sort by order of magnitude (largest first)
        sorted_units = sorted(
            available_units,
            key=lambda x: cls.get_oom_of_unit(x),
            reverse=True
        )

        # Find unit where most values will be < 10,000
        # Start with unit closest to median, then adjust if needed
        best_unit = cls.get_closest_unit_for_oom(base_unit, median_magnitude)

        # Verify that most values (>= 80%) fit well in this unit (< 10,000)
        values_in_unit = [abs(q.to(best_unit).magnitude) for q in quantities]
        non_zero_values = [v for v in values_in_unit if v > 0]

        if non_zero_values:
            fit_count = sum(1 for v in non_zero_values if v < 10_000)
            fit_ratio = fit_count / len(non_zero_values)

            # If less than 80% fit, try larger unit
            if fit_ratio < 0.8:
                unit_index = sorted_units.index(best_unit)
                if unit_index > 0:  # There's a larger unit available
                    best_unit = sorted_units[unit_index - 1]

        return best_unit

    @classmethod
    def get_all_units_with_equal_base(cls, unit: Unit) -> list[Unit]:
        return [u for u in Units if cls.units_have_same_base(unit, u)]

    @classmethod
    def get_pretty_text_for_quantity(
            cls,
            quantity: Quantity,
            decimals: int = None,
            thousands_separator: str = None,
            include_unit: bool = True,
            include_oom: bool = True,
            include_sign: bool = None,
    ) -> str:
        if decimals is None:
            decimals = cls.get_pretty_decimals(quantity)
        if thousands_separator is None:
            thousands_separator = ''

        sign_str = cls._get_sign_str_for_quantity(quantity, include_sign)
        value_str = f'{abs(quantity.magnitude):,.{decimals}f}'
        value_str = value_str.replace(',', thousands_separator)

        if include_unit:
            if not include_oom:
                raise NotImplementedError('Why would you do that?')
            unit_str = str(quantity.units)
        elif include_oom:
            unit_str = cls._get_units_oom_prefix(quantity.units)
        else:
            unit_str = ''

        components = []
        if sign_str:
            components.append(sign_str)

        components.append(value_str)

        if unit_str:
            components.append(' ' + unit_str)

        pretty_text = ''.join(components)

        for r, v in cls._STRING_REPLACEMENTS.items():
            pretty_text = pretty_text.replace(r, v)

        return pretty_text

    @classmethod
    def _get_sign_str_for_quantity(cls, quantity: Quantity, include_sign: bool = None) -> str:
        if include_sign is False:
            return ''

        value = quantity.magnitude
        if np.isnan(value):
            return ''
        if value == 0:
            return ''
        if value < 0:
            return '-'
        if value > 0:
            if include_sign:
                return '+'
            else:
                return ''
        raise Exception(f'How did you end up here for value {quantity}')

    @classmethod
    def get_pretty_decimals(cls, quantity: Quantity) -> int:

        # if quantity.units == Units.per_unit:
        #     return 3

        if isinstance(quantity.magnitude, int):
            return 0

        abs_value = abs(quantity.magnitude)
        if abs_value > 100:
            return 0
        elif abs_value > 10:
            return 1
        elif abs_value > 0.1:
            return 2
        elif abs_value > 0.01:
            return 3
        else:
            return 5

    @classmethod
    def _get_units_oom_prefix(cls, unit: Unit) -> str:
        base_unit = cls.get_base_unit_for_unit(unit)
        return str(unit).replace(str(base_unit), '')


class QuantityToTextConverter:
    """
    Configurable converter for formatting Quantity objects as text strings.

    Stores formatting configuration that can be reused across multiple quantity
    conversions, enabling consistent formatting across KPI collections and visualizations.

    Args:
        target_unit: Target unit for conversion (if None, uses pretty unit selection)
        decimals: Number of decimal places (if None, auto-determined)
        thousands_separator: Separator for thousands (default: '')
        include_unit: Whether to include unit in output (default: True)
        include_oom: Whether to include order of magnitude prefix (default: True)
        include_sign: Whether to include + sign for positive values (default: None/auto)

    Examples:
        Basic usage with fixed configuration:
        >>> converter = QuantityToTextConverter(
        ...     target_unit=Units.MWh,
        ...     decimals=2,
        ...     thousands_separator=' '
        ... )
        >>> converter.convert(5432.1 * Units.kWh)
        '5.43 MWh'

        Auto-configure from collection of quantities:
        >>> quantities = [1000 * Units.EUR, 5000 * Units.EUR, 10000 * Units.EUR]
        >>> converter = QuantityToTextConverter.from_quantities(quantities, decimals=0)
        >>> [converter.convert(q) for q in quantities]
        ['1 kEUR', '5 kEUR', '10 kEUR']
    """

    def __init__(
        self,
        target_unit: Unit = None,
        decimals: int = None,
        thousands_separator: str = None,
        include_unit: bool = True,
        include_oom: bool = True,
        include_sign: bool = None,
    ):
        self.target_unit = target_unit
        self.decimals = decimals
        self.thousands_separator = thousands_separator if thousands_separator is not None else ''
        self.include_unit = include_unit
        self.include_oom = include_oom
        self.include_sign = include_sign

    def convert(self, quantity: Quantity) -> str:
        """
        Convert a Quantity to formatted text string using stored configuration.

        Args:
            quantity: The quantity to format

        Returns:
            Formatted text representation
        """
        # Apply target unit conversion if specified
        if self.target_unit is not None:
            quantity = Units.get_quantity_in_target_unit(quantity, self.target_unit)
        else:
            quantity = Units.get_quantity_in_pretty_unit(quantity)

        # Use Units.get_pretty_text_for_quantity with stored configuration
        return Units.get_pretty_text_for_quantity(
            quantity,
            decimals=self.decimals,
            thousands_separator=self.thousands_separator,
            include_unit=self.include_unit,
            include_oom=self.include_oom,
            include_sign=self.include_sign,
        )

    @classmethod
    def from_quantities(
        cls,
        quantities: list[Quantity],
        target_unit: Unit = None,
        decimals: int = None,
        thousands_separator: str = None,
        include_unit: bool = True,
        include_oom: bool = True,
        include_sign: bool = None,
    ) -> 'QuantityToTextConverter':
        """
        Create converter auto-configured for a collection of quantities.

        Analyzes the provided quantities to determine an appropriate common unit
        (if target_unit not specified) that works well for all values.

        Args:
            quantities: Collection of quantities to analyze
            target_unit: Override auto-selected unit with specific target
            decimals: Number of decimal places (if None, will be auto-determined per value)
            thousands_separator: Separator for thousands (default: '')
            include_unit: Whether to include unit in output (default: True)
            include_oom: Whether to include order of magnitude prefix (default: True)
            include_sign: Whether to include + sign for positive values (default: None/auto)

        Returns:
            Configured QuantityToTextConverter instance

        Examples:
            Auto-configure for price data:
            >>> prices = [45.2 * Units.EUR_per_MWh, 67.8 * Units.EUR_per_MWh]
            >>> converter = QuantityToTextConverter.from_quantities(prices, thousands_separator=' ')
            >>> converter.convert(prices[0])
            '45.20 €/MWh'
        """
        # Determine common pretty unit if not explicitly provided
        if target_unit is None and quantities:
            target_unit = Units.get_common_pretty_unit_for_quantities(quantities)

        return cls(
            target_unit=target_unit,
            decimals=decimals,
            thousands_separator=thousands_separator,
            include_unit=include_unit,
            include_oom=include_oom,
            include_sign=include_sign,
        )


if __name__ == '__main__':
    test_values = [0.0123, 1.234, 1234.5678, 12345678.90123]
    test_units = [
        Units.Wh,
        Units.MWh,
        Units.GWh,
        Units.MW,
        Units.GW,
        Units.EUR,
        Units.EUR_per_MWh,
        Units.per_unit,
        Units.percent
    ]
    for uu in test_units:
        for vv in test_values:
            q = vv * uu
            qq = Units.get_quantity_in_pretty_unit(q)
            print(Units.get_pretty_text_for_quantity(qq))
