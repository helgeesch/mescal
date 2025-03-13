import pandas as pd
import numpy as np


class VolumeWeightedPriceAggregator:
    """Computes volume-weighted prices for regions based on bidding zone data.

    For each timestamp, the volume weights are determined by:
    1. If any bidding zone in the region has demand > 0: use matched demand as weights
    2. If no demand but some supply > 0: use matched supply as weights
    3. If no demand or supply: use simple average (equal weights)

    Node filtering (e.g., for virtual nodes only) should be done before passing
    the node_model_df to this class (e.g. pass node_model_df.query('is_virtual == True').

    Example:
        >>> # Filter nodes before creating aggregator
        >>> filtered_nodes = node_model_df.query('is_virtual == True')
        >>> aggregator = VolumeWeightedPriceAggregator(
        ...     node_model_df=filtered_nodes,
        ...     agg_region_column="country"
        ... )
        >>> region_prices = aggregator.aggregate_prices(price_df, matched_demand_df)
    """

    def __init__(
            self,
            node_model_df: pd.DataFrame,
            agg_region_column: str = "country"
    ):
        self.node_model_df = node_model_df.copy()
        self.agg_region_column = agg_region_column

    def _create_node_to_region_map(self) -> dict:
        return self.node_model_df[self.agg_region_column].to_dict()

    def get_all_regions(self) -> set:
        return set(self.node_model_df[self.agg_region_column].unique())

    def get_region_nodes(self, region: str) -> list:
        return self.node_model_df[self.node_model_df[self.agg_region_column] == region].index.to_list()

    def aggregate_prices(
            self,
            price_df: pd.DataFrame,
            matched_demand_df: pd.DataFrame = None,
            matched_supply_df: pd.DataFrame = None
    ) -> pd.DataFrame:
        self._check_input(price_df, matched_demand_df, matched_supply_df)
        result_dict = {}

        for region in self.get_all_regions():
            region_nodes = self.get_region_nodes(region)
            prices = price_df[region_nodes]
            weights = self._compute_volume_weights(prices, matched_demand_df, matched_supply_df)
            region_price = (prices * weights).sum(axis=1)
            result_dict[region] = region_price

        result_df = pd.DataFrame(result_dict)
        result_df.columns.name = self.agg_region_column
        return result_df

    def _compute_volume_weights(self, prices, matched_demand_df, matched_supply_df):
        region_nodes = prices.columns
        num_nodes = len(region_nodes)

        weights = pd.DataFrame(1, index=prices.index, columns=region_nodes, dtype=float)

        if matched_demand_df is not None:
            demand = matched_demand_df[region_nodes]
            has_demand = demand.sum(axis=1) > 0
            weights.loc[has_demand] = demand.loc[has_demand]

        if matched_supply_df is not None:
            supply = matched_supply_df[region_nodes]
            has_supply = supply.sum(axis=1) > 0
            if matched_demand_df is not None:
                # Only use supply where we don't have demand
                has_supply = has_supply & ~has_demand
            weights.loc[has_supply] = supply.loc[has_supply]

        weights = weights.div(weights.sum(axis=1), axis=0).fillna(1 / num_nodes)
        return weights

    def _check_input(
            self,
            price_df: pd.DataFrame,
            matched_demand_df: pd.DataFrame | None,
            matched_supply_df: pd.DataFrame | None
    ):
        if not isinstance(price_df, pd.DataFrame):
            raise TypeError("price_df must be a pandas DataFrame")

        relevant_nodes = set(self.node_model_df.index)
        price_nodes = set(price_df.columns)
        if not relevant_nodes.issubset(price_nodes):
            missing = relevant_nodes - price_nodes
            raise ValueError(f"Missing nodes in price_df: {missing}")

        for df, name in [
            (matched_demand_df, "matched_demand_df"),
            (matched_supply_df, "matched_supply_df")
        ]:
            if df is None:
                continue
            if not isinstance(df, pd.DataFrame):
                raise TypeError(f"{name} must be a pandas DataFrame")
            if not df.index.equals(price_df.index):
                raise ValueError(f"{name} must have same index as price_df")
            if not set(df.columns).issubset(price_df.columns):
                missing = relevant_nodes - set(df.columns)
                raise ValueError(f"Missing nodes in {name}: {missing}")


if __name__ == "__main__":
    # Create dummy data
    index = pd.date_range("2024-01-01", periods=24, freq="h")
    zones = ["DE1", "DE2", "FR1", "BE1"]

    # Random prices between 20 and 100
    prices = pd.DataFrame(
        np.random.uniform(20, 100, size=(24, len(zones))),
        index=index,
        columns=zones
    )

    # Random demand between 0 and 1000, some hours with zero demand
    demands = pd.DataFrame(
        np.random.uniform(0, 1000, size=(24, len(zones))),
        index=index,
        columns=zones
    )
    demands.iloc[0:3] = 0  # First three hours no demand

    # Random supply between 0 and 1000, some hours with zero supply
    supplies = pd.DataFrame(
        np.random.uniform(0, 1000, size=(24, len(zones))),
        index=index,
        columns=zones
    )
    supplies.iloc[0] = 0  # First hour no supply

    # Create node mapping and filter before creating aggregator
    node_model_df = pd.DataFrame({
        "country": ["DE", "DE", "FR", "BE"],
        "is_virtual": [True, False, True, False]
    }, index=zones)

    # Filter nodes
    virtual_nodes = node_model_df.query("is_virtual == True")

    # Initialize aggregator with pre-filtered nodes
    aggregator = VolumeWeightedPriceAggregator(
        node_model_df=virtual_nodes,
        agg_region_column="country"
    )

    # Test different combinations
    prices_with_demand = aggregator.aggregate_prices(prices, matched_demand_df=demands)
    prices_with_supply = aggregator.aggregate_prices(prices, matched_supply_df=supplies)
    prices_with_both = aggregator.aggregate_prices(prices, demands, supplies)
    prices_without_volumes = aggregator.aggregate_prices(prices)

    print("\nPrices with demand weights:\n", prices_with_demand.head())
    print("\nPrices with supply weights:\n", prices_with_supply.head())
    print("\nPrices with both weights:\n", prices_with_both.head())
    print("\nPrices without volume weights (simple average):\n", prices_without_volumes.head())