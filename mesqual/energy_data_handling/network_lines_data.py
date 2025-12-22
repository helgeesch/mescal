from dataclasses import dataclass

import pandas as pd


class NetworkLineFlowsData:
    """Wrapper for bidirectional flow data of network transmission lines.

    This class encapsulates energy or power flow data for lines in both directions,
    accounting for transmission losses. It provides a standardized interface for
    handling complex flow patterns in electrical network analysis.

    Flow Direction Conventions:
        - sent_up: Flow entering line at node_from (towards node_to)
        - received_up: Flow leaving line at node_from after losses (coming from node_to)
        - sent_down: Flow entering line at node_to (towards node_from)  
        - received_down: Flow leaving line at node_to after losses (coming from node_from)

    The distinction between 'sent' and 'received' allows for modeling transmission
    losses, where received_flow = sent_flow * (1 - loss_rate).

    Args:
        sent_up: DataFrame with flow data entering at node_from
        received_up: DataFrame with flow data received at node_from after losses
        sent_down: DataFrame with flow data entering at node_to
        received_down: DataFrame with flow data received at node_to after losses
        granularity: Time granularity of the data (None, float in minutes, or Series)

    Raises:
        ValueError: If indices or columns of the four DataFrames don't match

    Example:

        >>> import pandas as pd
        >>> index = pd.date_range('2024-01-01', periods=24, freq='1H')
        >>> columns = ['Line_A_B', 'Line_B_C']
        >>> flows_up = pd.DataFrame(100, index=index, columns=columns)
        >>> flows_down = pd.DataFrame(50, index=index, columns=columns)
        >>> line_data = NetworkLineFlowsData.from_up_and_down_flow_without_losses(
        ...     flows_up, flows_down)
    """
    def __init__(
            self,
            sent_up: pd.DataFrame,
            received_up: pd.DataFrame,
            sent_down: pd.DataFrame,
            received_down: pd.DataFrame,
            granularity: None | float | pd.Series = None
    ):
        """Initialize NetworkLineFlowsData with flow data in both directions.
        
        Args:
            sent_up: Flow data sent in up direction (node_from -> node_to)
            received_up: Flow data received in up direction after losses
            sent_down: Flow data sent in down direction (node_to -> node_from)
            received_down: Flow data received in down direction after losses
            granularity: Time granularity information for the data
        """
        self.sent_up = sent_up
        self.received_up = received_up
        self.sent_down = sent_down
        self.received_down = received_down
        self.granularity = granularity
        self.__post_init__()

    def __post_init__(self):
        """Validate that all DataFrames have matching indices and columns.
        
        Raises:
            ValueError: If any DataFrame has mismatched indices or columns.
        """
        dataframes = [self.received_up, self.sent_down, self.received_down]
        for i, df in enumerate(dataframes):
            df_name = ['received_up', 'sent_down', 'received_down'][i]
            if not self.sent_up.index.equals(df.index):
                raise ValueError(f'Index mismatch: sent_up vs {df_name}')
            if not self.sent_up.columns.equals(df.columns):
                raise ValueError(f'Columns mismatch: sent_up vs {df_name}')

    def from_mw_to_mwh(self) -> 'NetworkLineFlowsData':
        """Convert flow data from MW (power) to MWh (energy).
        
        This conversion requires granularity information to properly scale the values.
        
        Returns:
            New NetworkLineFlowsData instance with energy values
            
        Raises:
            NotImplementedError: Method not yet implemented
        """
        # TODO: Implement MW to MWh conversion using granularity
        raise NotImplementedError("MW to MWh conversion not yet implemented")

    def from_mwh_to_mw(self) -> 'NetworkLineFlowsData':
        """Convert flow data from MWh (energy) to MW (power).
        
        This conversion requires granularity information to properly scale the values.
        
        Returns:
            New NetworkLineFlowsData instance with power values
            
        Raises:
            NotImplementedError: Method not yet implemented
        """
        # TODO: Implement MWh to MW conversion using granularity  
        raise NotImplementedError("MWh to MW conversion not yet implemented")

    @classmethod
    def from_net_flow_without_losses(cls, net_flow: pd.DataFrame) -> "NetworkLineFlowsData":
        """Create NetworkLineFlowsData from net line flow data assuming no transmission losses.
        
        Converts net flow data (where positive values indicate flow in up direction
        and negative values indicate flow in down direction) into the bidirectional
        flow representation used by this class.
        
        Args:
            net_flow: DataFrame with net flow values. Positive = up direction,
                negative = down direction
                
        Returns:
            NetworkLineFlowsData instance with flows split into up/down directions
            
        Example:
            
            >>> import pandas as pd
            >>> net_flows = pd.DataFrame({
            ...     'Line_A_B': [100, -50, 75],
            ...     'Line_B_C': [200, 150, -100]
            ... })
            >>> line_data = NetworkLineFlowsData.from_net_flow_without_losses(net_flows)
        """
        positive_flow = net_flow.clip(lower=0)
        negative_flow = -net_flow.clip(upper=0)

        return cls(
            sent_up=positive_flow,
            received_up=positive_flow,
            sent_down=negative_flow,
            received_down=negative_flow
        )

    @classmethod
    def from_nodal_net_injection(
            cls,
            node_a_net_injection: pd.DataFrame,
            node_b_net_injection: pd.DataFrame
    ) -> "NetworkLineFlowsData":
        """Create NetworkLineFlowsData from nodal net-injection data.

        Converts nodal net-injection data (where positive values indicate flow from node towards line (injection)
        and negative values indicate flow from line towards node (ejection)) into the bidirectional
        flow representation used by this class. Automatically computes losses.
        Node_A refers to the topological "node_from" and
        Node_B to the topological "node_to" in the definition of the lines.

        Args:
            node_a_net_injection: DataFrame with net injection at Node_A.
                positive = injection, negative = ejection
            node_b_net_injection: DataFrame with net injection at Node_B.
                positive = injection, negative = ejection

        Returns:
            NetworkLineFlowsData instance with flows split into up/down directions

        Example:

            >>> import pandas as pd
            >>> net_injection_a = pd.DataFrame({
            ...     'power_a': [100, -49, -49],
            ... })
            >>> net_injection_b = pd.DataFrame({
            ...     'power_b': [-98, 50,  50]
            ... })
            >>> line_data = NetworkLineFlowsData.from_nodal_net_injection(net_injection_a, net_injection_b)
        """

        return cls(
            sent_up=node_a_net_injection.clip(0),
            received_up=-1 * node_b_net_injection.clip(None, 0),
            sent_down=node_b_net_injection.clip(0),
            received_down=-1 * node_a_net_injection.clip(None, 0)
        )

    @classmethod
    def from_up_and_down_flow_without_losses(
            cls,
            flow_up: pd.DataFrame,
            flow_down: pd.DataFrame
    ) -> "NetworkLineFlowsData":
        """Create NetworkLineFlowsData from separate up and down flow data without losses.
        
        This constructor assumes that there are no transmission losses, so sent and
        received flows are identical in each direction.
        
        Args:
            flow_up: DataFrame with flow data in up direction (node_from -> node_to)
            flow_down: DataFrame with flow data in down direction (node_to -> node_from)
            
        Returns:
            NetworkLineFlowsData instance where sent and received flows are equal
            
        Example:
            
            >>> import pandas as pd
            >>> up_flows = pd.DataFrame({'Line_A_B': [100, 80, 120]})
            >>> down_flows = pd.DataFrame({'Line_A_B': [50, 60, 40]})
            >>> line_data = NetworkLineFlowsData.from_up_and_down_flow_without_losses(
            ...     up_flows, down_flows)
        """
        return cls(
            sent_up=flow_up,
            received_up=flow_up,
            sent_down=flow_down,
            received_down=flow_down
        )


@dataclass
class NetworkLineCapacitiesData:
    """Wrapper for bidirectional capacity data of network transmission lines.

    This dataclass encapsulates transmission capacity limits for network lines in both
    directions. Capacities can be asymmetric to reflect real-world transmission
    constraints or operational limits.

    Capacity Direction Conventions:
        - capacities_up: Maximum transmission capacity from node_from to node_to
        - capacities_down: Maximum transmission capacity from node_to to node_from

    Args:
        capacities_up: DataFrame with capacity limits in up direction
        capacities_down: DataFrame with capacity limits in down direction  
        granularity: Time granularity of capacity data (None, float in minutes, or Series)

    Raises:
        ValueError: If indices or columns of the two DataFrames don't match

    Example:

        >>> import pandas as pd
        >>> index = pd.date_range('2024-01-01', periods=24, freq='1H')
        >>> columns = ['Line_A_B', 'Line_B_C']
        >>> caps = pd.DataFrame(1000, index=index, columns=columns)
        >>> capacity_data = NetworkLineCapacitiesData.from_symmetric_capacities(caps)
    """
    capacities_up: pd.DataFrame
    capacities_down: pd.DataFrame
    granularity: None | float | pd.Series = None

    def __post_init__(self):
        """Validate that both capacity DataFrames have matching indices and columns.
        
        Raises:
            ValueError: If DataFrames have mismatched indices or columns.
        """
        if not self.capacities_up.index.equals(self.capacities_down.index):
            raise ValueError('Index mismatch: capacities_up vs capacities_down')
        if not self.capacities_up.columns.equals(self.capacities_down.columns):
            raise ValueError('Columns mismatch: capacities_up vs capacities_down')

    def from_mw_to_mwh(self) -> 'NetworkLineCapacitiesData':
        """Convert capacity data from MW (power) to MWh (energy).
        
        This conversion requires granularity information to properly scale the values.
        
        Returns:
            New NetworkLineCapacitiesData instance with energy capacity values
            
        Raises:
            NotImplementedError: Method not yet implemented
        """
        # TODO: Implement MW to MWh conversion using granularity
        raise NotImplementedError("MW to MWh conversion not yet implemented")

    def from_mwh_to_mw(self) -> 'NetworkLineCapacitiesData':
        """Convert capacity data from MWh (energy) to MW (power).
        
        This conversion requires granularity information to properly scale the values.
        
        Returns:
            New NetworkLineCapacitiesData instance with power capacity values
            
        Raises:
            NotImplementedError: Method not yet implemented
        """
        # TODO: Implement MWh to MW conversion using granularity
        raise NotImplementedError("MWh to MW conversion not yet implemented")

    @classmethod
    def from_symmetric_capacities(cls, capacities: pd.DataFrame) -> "NetworkLineCapacitiesData":
        """Create NetworkLineCapacitiesData with identical capacities in both directions.
        
        This is a convenience constructor for cases where transmission lines have
        the same capacity limit in both directions.
        
        Args:
            capacities: DataFrame with capacity values to use for both directions
            
        Returns:
            NetworkLineCapacitiesData instance with symmetric capacities
            
        Example:
            
            >>> import pandas as pd
            >>> caps = pd.DataFrame({
            ...     'Line_A_B': [1000, 1200, 800],
            ...     'Line_B_C': [1500, 1500, 1000]
            ... })
            >>> capacity_data = NetworkLineCapacitiesData.from_symmetric_capacities(caps)
        """
        return cls(capacities_up=capacities, capacities_down=capacities)


if __name__ == '__main__':
    import numpy as np
    
    # Create sample data for demonstration
    index = pd.date_range('2024-01-01', periods=24, freq='1H')
    columns = ['Line_A_B', 'Line_B_C', 'Line_C_D']
    
    # Test NetworkLineFlowsData
    print("=== NetworkLineFlowsData Examples ===")
    
    # Example 1: From net flow data
    net_flows = pd.DataFrame({
        'Line_A_B': np.random.normal(100, 30, 24),
        'Line_B_C': np.random.normal(-50, 20, 24),
        'Line_C_D': np.random.normal(75, 25, 24)
    }, index=index)
    
    flow_data1 = NetworkLineFlowsData.from_net_flow_without_losses(net_flows)
    print(f"From net flows - sent_up shape: {flow_data1.sent_up.shape}")
    print(f"Sample sent_up data:\n{flow_data1.sent_up.head(3)}")
    print(f"Sample sent_down data:\n{flow_data1.sent_down.head(3)}")
    
    # Example 2: From separate up/down flows
    flows_up = pd.DataFrame(np.random.exponential(80, (24, 3)), 
                           index=index, columns=columns)
    flows_down = pd.DataFrame(np.random.exponential(60, (24, 3)),
                             index=index, columns=columns)
    
    flow_data2 = NetworkLineFlowsData.from_up_and_down_flow_without_losses(
        flows_up, flows_down)
    print(f"\nFrom up/down flows - received_up equals sent_up: {flow_data2.sent_up.equals(flow_data2.received_up)}")
    
    # Test NetworkLineCapacitiesData
    print("\n=== NetworkLineCapacitiesData Examples ===")
    
    # Example 1: Symmetric capacities
    capacities = pd.DataFrame({
        'Line_A_B': [1000, 1200, 800, 1100] * 6,
        'Line_B_C': [1500, 1300, 1400, 1600] * 6,
        'Line_C_D': [900, 1000, 950, 1050] * 6
    }, index=index)
    
    capacity_data = NetworkLineCapacitiesData.from_symmetric_capacities(capacities)
    print(f"Symmetric capacities - up equals down: {capacity_data.capacities_up.equals(capacity_data.capacities_down)}")
    print(f"Sample capacity data:\n{capacity_data.capacities_up.head(3)}")
    
    # Example 2: Asymmetric capacities
    capacities_up = capacities * 1.1  # 10% higher in up direction
    capacities_down = capacities * 0.9  # 10% lower in down direction
    
    capacity_data_asym = NetworkLineCapacitiesData(
        capacities_up=capacities_up,
        capacities_down=capacities_down
    )
    print(f"\nAsymmetric capacities - different up/down: {not capacity_data_asym.capacities_up.equals(capacity_data_asym.capacities_down)}")
