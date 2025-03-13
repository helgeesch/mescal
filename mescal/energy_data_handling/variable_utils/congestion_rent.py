from dataclasses import dataclass
import pandas as pd

from mescal.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CongestionRentCalculator:
    sent_up: pd.Series
    received_up: pd.Series
    sent_down: pd.Series
    received_down: pd.Series
    price_node_from: pd.Series
    price_node_to: pd.Series
    granularity_hrs: float = None

    def __post_init__(self):
        if isinstance(self.sent_up.index, pd.DatetimeIndex):
            from mescal.energy_data_handling.granularity_analyzer import TimeSeriesGranularityAnalyzer
            analyzer = TimeSeriesGranularityAnalyzer(strict_mode=False)
            self.granularity_hrs = analyzer.get_granularity_as_hours(self.sent_up.index)
        else:
            if self.granularity_hrs is None:
                logger.warning(f'Granularity for CongestionRentCalculator is defaulting back to 1 hrs.')
        self.__check_indices()

    def __check_indices(self):
        ref_index = self.sent_up.index
        to_check = [
            self.received_up,
            self.sent_down,
            self.received_down,
            self.price_node_from,
            self.price_node_to
        ]
        for v in to_check:
            if not ref_index.equals(v.index):
                raise ValueError(f'All indices of provided series must be equal.')

    @property
    def congestion_rent_up(self) -> pd.Series:
        return self.granularity_hrs * (
                self.received_up * self.price_node_to -
                self.sent_up * self.price_node_from
        )

    @property
    def congestion_rent_down(self) -> pd.Series:
        return self.granularity_hrs * (
                self.received_down * self.price_node_from -
                self.sent_down * self.price_node_to
        )

    @property
    def congestion_rent_total(self) -> pd.Series:
        return self.congestion_rent_up + self.congestion_rent_down

    def calculate(self) -> pd.Series:
        return self.congestion_rent_total

    @classmethod
    def from_net_flow_without_losses(
            cls,
            net_flow: pd.Series,
            price_node_from: pd.Series,
            price_node_to: pd.Series,
            granularity_hrs: float = None
    ) -> pd.Series:
        sent_up = net_flow.clip(lower=0)
        sent_down = (-net_flow).clip(lower=0)
        return cls(
            sent_up=sent_up,
            received_up=sent_up,
            sent_down=sent_down,
            received_down=sent_down,
            price_node_from=price_node_from,
            price_node_to=price_node_to,
            granularity_hrs=granularity_hrs
        ).calculate()

    @classmethod
    def from_up_and_down_flow_without_losses(
            cls,
            flow_up: pd.Series,
            flow_down: pd.Series,
            price_node_from: pd.Series,
            price_node_to: pd.Series,
            granularity_hrs: float = None
    ) -> pd.Series:
        return cls(
            sent_up=flow_up,
            received_up=flow_up,
            sent_down=flow_down,
            received_down=flow_down,
            price_node_from=price_node_from,
            price_node_to=price_node_to,
            granularity_hrs=granularity_hrs
        ).calculate()


if __name__ == "__main__":
    index = pd.date_range("2024-01-01", periods=24, freq="H")
    dummy_data = {
        "sent_up": pd.Series([100, 150, 200], index=index[:3]),
        "received_up": pd.Series([95, 142, 190], index=index[:3]),
        "sent_down": pd.Series([50, 75, 100], index=index[:3]),
        "received_down": pd.Series([48, 71, 95], index=index[:3]),
        "price_node_from": pd.Series([45, 50, 55], index=index[:3]),
        "price_node_to": pd.Series([65, 70, 75], index=index[:3])
    }

    calculator = CongestionRentCalculator(**dummy_data)

    print("Direction-wise congestion rents:")
    print("Up direction:", calculator.congestion_rent_up)
    print("\nDown direction:", calculator.congestion_rent_down)
    print("\nTotal:", calculator.congestion_rent_total)

    net_flow = pd.Series([50, 75, 100], index=index[:3])
    simple_rent = CongestionRentCalculator.from_net_flow_without_losses(
        net_flow=net_flow,
        price_node_from=dummy_data["price_node_from"],
        price_node_to=dummy_data["price_node_to"]
    )
    print("\nSimplified calculation without losses:")
    print(simple_rent)
