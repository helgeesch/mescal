from typing import Iterator, ClassVar
import numpy as np
from pint import get_application_registry, UnitRegistry, set_application_registry, Unit, Quantity

from mescal.enums import QuantityTypeEnum


class UnitNotFound(Exception):
    pass


class UnitRegistryNotComplete(Exception):
    def __init__(self, message: str = None):
        base = f'You should never end up here. Your units are not properly registered in the {Units.__name__} class.'
        message = message or ''
        super().__init__(base + message)


ureg = UnitRegistry()

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
            decimals = cls._get_pretty_decimals(quantity)
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
    def _get_pretty_decimals(cls, quantity: Quantity) -> int:

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
