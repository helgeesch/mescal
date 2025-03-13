from dataclasses import dataclass
from enum import Enum
import pandas as pd
import networkx as nx


class FlowType(Enum):
    PRE_LOSS = "pre_loss"
    POST_LOSS = "post_loss"


# TODO: use also in CongestionRent
@dataclass
class LineFlowData:
    sent_up: pd.DataFrame
    received_up: pd.DataFrame
    sent_down: pd.DataFrame
    received_down: pd.DataFrame

    def __post_init__(self):
        for s in [self.received_up, self.sent_down, self.received_down]:
            if not self.sent_up.index.equals(s.index):
                raise ValueError(f'All indices must be equal!')

    def from_mw_to_mwh(self) -> 'LineFlowData':
        # TODO
        raise NotImplementedError

    def from_mwh_to_mw(self) -> 'LineFlowData':
        # TODO
        raise NotImplementedError

    @classmethod
    def from_net_flow_without_losses(cls, net_flow: pd.DataFrame) -> "LineFlowData":
        positive_flow = net_flow.clip(lower=0)
        negative_flow = -net_flow.clip(upper=0)

        return cls(
            sent_up=positive_flow,
            received_up=positive_flow,
            sent_down=negative_flow,
            received_down=negative_flow
        )

    @classmethod
    def from_up_and_down_flow_without_losses(
            cls,
            flow_up: pd.DataFrame,
            flow_down: pd.DataFrame
    ) -> "LineFlowData":
        return cls(
            sent_up=flow_up,
            received_up=flow_up,
            sent_down=flow_down,
            received_down=flow_down
        )


class RegionalTradeBalanceCalculator:
    """Aggregates bidirectional power flows between regions based on node-level flow data.

    Takes line-level flow data with bidirectional flows (up/down) and losses, and aggregates
    them to a higher regional level (e.g., countries, market areas) or keeps them at node level.
    Uses networkx for identifying region/node connections and handles multiple lines between
    the same region/node pairs.

    When agg_region_column is None, each node is treated as its own region, allowing for
    node-to-node trade balance analysis without aggregation.

    Example for regional aggregation:
        >>> line_model_df = pd.DataFrame({
        ...     "node_from": ["DE1", "FR1"],
        ...     "node_to": ["FR1", "BE1"]
        ... })
        >>> node_model_df = pd.DataFrame({
        ...     "country": ["DE", "FR", "BE"]
        ... }, index=["DE1", "FR1", "BE1"])
        >>> aggregator = RegionalTradeBalanceCalculator(
        ...     line_model_df=line_model_df,
        ...     node_model_df=node_model_df,
        ...     agg_region_column="country"
        ... )

    Example for node-level flows:
        >>> aggregator = RegionalTradeBalanceCalculator(
        ...     line_model_df=line_model_df,
        ...     node_model_df=node_model_df,
        ...     agg_region_column=None  # Keep flows at node level
        ... )
    """
    EXP_VAR = "exp"
    IMP_VAR = "imp"
    NET_EXP_VAR = "net_exp"
    ALL_VARS = [EXP_VAR, IMP_VAR, NET_EXP_VAR]

    def __init__(
            self,
            line_model_df: pd.DataFrame,
            node_model_df: pd.DataFrame,
            agg_region_column: str | None = "country",
            node_from_col: str = "node_from",
            node_to_col: str = "node_to"
    ):
        self.line_model_df = line_model_df
        self.node_model_df = node_model_df
        self.agg_region_column = agg_region_column
        self.node_from_col = node_from_col
        self.node_to_col = node_to_col
        self.node_to_agg_region_map = self._create_node_to_region_map()
        self.agg_region_graph = self._create_region_graph()

    def _create_node_to_region_map(self) -> dict:
        if self.agg_region_column is None:
            return {node: node for node in self.node_model_df.index}
        return self.node_model_df[self.agg_region_column].to_dict()

    def _create_region_graph(self) -> nx.Graph:
        graph = nx.Graph()

        for _, line in self.line_model_df.iterrows():
            region_from = self.node_to_agg_region_map[line[self.node_from_col]]
            region_to = self.node_to_agg_region_map[line[self.node_to_col]]

            if region_from != region_to:
                if not graph.has_edge(region_from, region_to):
                    graph.add_edge(region_from, region_to)

        return graph

    def _get_net_exp_for_couple(self, primary, secondary, flow_data: LineFlowData, flow_type: FlowType) -> pd.Series:
        mask_forward = (
                (self.line_model_df[self.node_from_col].map(self.node_to_agg_region_map) == primary) &
                (self.line_model_df[self.node_to_col].map(self.node_to_agg_region_map) == secondary)
        )
        mask_backward = (
                (self.line_model_df[self.node_from_col].map(self.node_to_agg_region_map) == secondary) &
                (self.line_model_df[self.node_to_col].map(self.node_to_agg_region_map) == primary)
        )

        lines_forward = self.line_model_df[mask_forward].index
        lines_backward = self.line_model_df[mask_backward].index

        if flow_type == FlowType.PRE_LOSS:
            return (
                    flow_data.sent_up[lines_forward].sum(axis=1) -
                    flow_data.sent_down[lines_forward].sum(axis=1) +
                    flow_data.sent_down[lines_backward].sum(axis=1) -
                    flow_data.sent_up[lines_backward].sum(axis=1)
            )
        else:  # POST_LOSS
            return (
                    flow_data.sent_up[lines_forward].sum(axis=1) -
                    flow_data.received_down[lines_forward].sum(axis=1) +
                    flow_data.sent_down[lines_backward].sum(axis=1) -
                    flow_data.received_up[lines_backward].sum(axis=1)
            )

    def get_trade_balance(
            self,
            flow_data: LineFlowData,
            flow_type: FlowType = FlowType.POST_LOSS
    ) -> pd.DataFrame:
        flows_list = []
        column_level_names = [self.primary_name, self.partner_name, "variable"]

        for primary in self.get_all_regions():
            for secondary in self.get_region_neighbors(primary):
                net_exp = self._get_net_exp_for_couple(primary, secondary, flow_data, flow_type)
                df = pd.concat(
                    {
                        (primary, secondary, self.NET_EXP_VAR): net_exp,
                        (primary, secondary, self.EXP_VAR): net_exp.clip(0),
                        (primary, secondary, self.IMP_VAR): net_exp.clip(None, 0).abs(),
                    },
                    axis=1,
                    names=column_level_names,
                )
                flows_list.append(df)

        if not flows_list:
            return pd.DataFrame(
                index=flow_data.sent_up.index,
                columns=pd.MultiIndex.from_tuples([], names=column_level_names)
            )

        return pd.concat(flows_list, axis=1)

    def get_region_neighbors(self, region: str) -> set:
        return set(self.agg_region_graph.neighbors(region))

    def get_all_regions(self) -> set:
        return set(self.agg_region_graph.nodes())

    @property
    def primary_name(self) -> str:
        return f"primary_{self.agg_region_column}"

    @property
    def partner_name(self) -> str:
        return f"partner_{self.agg_region_column}"

    def aggregate_trade_balance_to_primary_level(self, trade_balance_df: pd.DataFrame) -> pd.DataFrame:
        """Reduces three-level trade balance DataFrame to primary region and variable only."""
        if trade_balance_df.columns.nlevels != 3:
            raise ValueError("Input DataFrame must have three column levels")
        if trade_balance_df.columns.names != [self.primary_name, self.partner_name, "variable"]:
            raise ValueError("Input DataFrame must be in format from aggregate_flows")

        return trade_balance_df.T.groupby(level=[self.primary_name, "variable"]).sum().T


if __name__ == "__main__":
    import numpy as np

    # Create dummy model data
    line_model_df = pd.DataFrame({
        "node_from": ["DE1", "DE1", "FR1", "BE1"],
        "node_to": ["FR1", "BE1", "BE1", "NL1"]
    })

    node_model_df = pd.DataFrame({
        "country": ["DE", "DE", "FR", "BE", "NL"],
        "macro_region": ["CWE", "CWE", "CWE", "CWE", "CWE"]
    }, index=["DE1", "DE2", "FR1", "BE1", "NL1"])

    # Create dummy flow data
    time_index = pd.date_range("2024-01-01", periods=24, freq="h")
    n_lines = len(line_model_df)

    # Create random raw flows - positive means up direction, negative means down direction
    raw_flows = pd.DataFrame(
        np.random.randn(24, n_lines) * 1000,  # random positive and negative values
        index=time_index,
        columns=range(n_lines)
    )

    # Split into up and down flows
    flow_up = raw_flows.clip(lower=0)  # only positive values, rest zero
    flow_down = -raw_flows.clip(upper=0)  # convert negative to positive, rest zero

    # Add 2% losses to received flows
    flow_data = LineFlowData(
        sent_up=flow_up,
        received_up=flow_up * 0.98,  # 2% losses
        sent_down=flow_down,
        received_down=flow_down * 0.98  # 2% losses
    )

    # Initialize aggregator and compute flows
    aggregator = RegionalTradeBalanceCalculator(
        line_model_df=line_model_df,
        node_model_df=node_model_df,
        agg_region_column="country"
    )

    # Get region flows
    country_flows_post_loss = aggregator.get_trade_balance(flow_data, FlowType.POST_LOSS)
    country_flows_pre_loss = aggregator.get_trade_balance(flow_data, FlowType.PRE_LOSS)

    print("\nNeighboring countries for DE:", aggregator.get_region_neighbors("DE"))
    print("\nAll countries:", aggregator.get_all_regions())
    print("\nCountry flows (post-loss):\n", country_flows_post_loss.head())
    print("\nCountry flows (pre-loss):\n", country_flows_pre_loss.head())