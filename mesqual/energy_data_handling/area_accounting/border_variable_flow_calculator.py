from typing import Literal

import numpy as np
import pandas as pd

from mesqual.energy_data_handling.network_lines_data import NetworkLineFlowsData
from mesqual.energy_data_handling.area_accounting.border_variable_base import AreaBorderVariableCalculatorBase


class BorderFlowCalculator(AreaBorderVariableCalculatorBase):
    """Calculates aggregated power flows for area borders.
    
    This calculator aggregates line-level power flows to border level, handling
    bidirectional flow data and transmission losses.
    The calculator can aggregate both sent and received flows, accounting for
    transmission losses that occur between sending and receiving ends. It supports
    multiple output formats including directional flows and net flows.
    
    Flow aggregation logic:
    - Lines and flows are classified as "up" or "down" based on topological direction
    - Flows are aggregated respecting directionality and loss conventions
    - Net flows represent the algebraic sum (up_flow - down_flow)
    
    Example:

        >>> from mesqual.energy_data_handling.network_lines_data import NetworkLineFlowsData
        >>> calculator = BorderFlowCalculator(
        ...     area_border_model_df, line_model_df, node_model_df, 'country'
        ... )
        >>> # Calculate net sent flows (before losses)
        >>> net_flows = calculator.calculate(flow_data, flow_type='sent', direction='net')
        >>> print(net_flows)
    """
    
    @property
    def variable_name(self) -> str:
        return "border_flow"
    
    def calculate(
        self,
        line_flow_data: NetworkLineFlowsData,
        flow_type: Literal['sent', 'received'] = 'sent',
        direction: Literal['up', 'down', 'net'] = 'net'
    ) -> pd.DataFrame:
        """Aggregate line-level power flows to border level.
        
        Sums power flows of all lines belonging to each border, respecting flow
        directionality and transmission loss conventions. The aggregation handles
        both pre-loss (sent) and post-loss (received) flows.
        
        Flow type selection:
        - 'sent': Flows before transmission losses (injected into lines)  
        - 'received': Flows after transmission losses (withdrawn from lines)
        
        Direction options:
        - 'up': Flows from area_from to area_to only
        - 'down': Flows from area_to to area_from only
        - 'net': Net flows (up - down), positive means net export from area_from
        
        The method handles missing data by preserving NaN values when all 
        constituent flows are missing for a given timestamp.
        
        Args:
            line_flow_data: NetworkLineFlowsData containing bidirectional flow
                time series. Must include sent_up, received_up, sent_down, and
                received_down DataFrames with line IDs as columns.
            flow_type: Type of flows to aggregate:
                - 'sent': Pre-loss flows (power injected into transmission)
                - 'received': Post-loss flows (power withdrawn after losses)
            direction: Flow direction to calculate:
                - 'up': Flows from area_from → area_to
                - 'down': Flows from area_to → area_from  
                - 'net': Net flows (up - down)
                
        Returns:
            DataFrame with border-level flow aggregations. Index matches input
            flow data, columns are border identifiers. Values represent power
            flows in MW. For net flows, positive values indicate net export
            from area_from to area_to.
            
        Raises:
            ValueError: If flow_type not in ['sent', 'received'] or direction
                not in ['up', 'down', 'net']
                
        Example:
            
            >>> # Calculate net sent flows (most common use case)
            >>> net_sent = calculator.calculate(flows, 'sent', 'net')
            >>> 
            >>> # Calculate received flows in up direction only
            >>> up_received = calculator.calculate(flows, 'received', 'up')
            >>> 
            >>> print(f"DE→FR net flow: {net_sent.loc['2024-01-01 12:00', 'DE-FR']:.0f} MW")
        """
        # Validate inputs
        if flow_type not in ['sent', 'received']:
            raise ValueError(f"Unknown flow_type: {flow_type}. Must be 'sent' or 'received'")
        if direction not in ['up', 'down', 'net']:
            raise ValueError(f"Unknown flow direction: {direction}. Must be 'up', 'down', or 'net'")
        
        self._validate_time_series_data(line_flow_data.sent_up, "sent_up")
        self._validate_time_series_data(line_flow_data.received_up, "received_up")
        
        border_flows = {}
        
        for border_id, border in self.area_border_model_df.iterrows():
            lines_up, lines_down = self.get_border_lines_in_topological_up_and_down_direction(border_id)

            if not lines_up and not lines_down:
                # No lines for this border - create empty series
                index = line_flow_data.sent_up.index
                border_flows[border_id] = pd.Series(index=index, dtype=float)
                continue

            # Select appropriate flow data based on flow_type
            if flow_type == 'sent':
                flow_data_up = line_flow_data.sent_up
                flow_data_down = line_flow_data.sent_down
            else:  # flow_type == 'received'
                flow_data_up = line_flow_data.received_up  
                flow_data_down = line_flow_data.received_down

            # Aggregate flows by direction relative to border
            flow_parts_up = []
            flow_parts_down = []
            
            if lines_up:
                # Lines in topological "up" direction
                available_lines_up = [line for line in lines_up if line in flow_data_up.columns]
                if available_lines_up:
                    flow_parts_up.append(flow_data_up[available_lines_up])
                    
            if lines_down:  
                # Lines in topological "down" direction contribute to opposite border flow
                available_lines_down = [line for line in lines_down if line in flow_data_down.columns]
                if available_lines_down:
                    flow_parts_up.append(flow_data_down[available_lines_down])
            
            if lines_down:
                # Lines in topological "down" direction  
                available_lines_down = [line for line in lines_down if line in flow_data_up.columns]
                if available_lines_down:
                    flow_parts_down.append(flow_data_up[available_lines_down])
                    
            if lines_up:
                # Lines in topological "up" direction contribute to opposite border flow
                available_lines_up = [line for line in lines_up if line in flow_data_down.columns]
                if available_lines_up:
                    flow_parts_down.append(flow_data_down[available_lines_up])

            # Sum flows for each direction
            if flow_parts_up:
                flows_up_combined = pd.concat(flow_parts_up, axis=1)
                flow_up = flows_up_combined.sum(axis=1)
                flow_up[flows_up_combined.isna().all(axis=1)] = np.nan
            else:
                flow_up = pd.Series(index=line_flow_data.sent_up.index, dtype=float)
                
            if flow_parts_down:
                flows_down_combined = pd.concat(flow_parts_down, axis=1)  
                flow_down = flows_down_combined.sum(axis=1)
                flow_down[flows_down_combined.isna().all(axis=1)] = np.nan
            else:
                flow_down = pd.Series(index=line_flow_data.sent_up.index, dtype=float)

            # Select final output based on direction parameter
            if direction == 'up':
                border_flows[border_id] = flow_up
            elif direction == 'down':
                border_flows[border_id] = flow_down
            else:  # direction == 'net'
                flow_net = flow_up.subtract(flow_down, fill_value=0)
                # Preserve NaN when both directions are NaN
                flow_net[flow_up.isna() & flow_down.isna()] = np.nan
                border_flows[border_id] = flow_net
        
        result = pd.DataFrame(border_flows)
        result.columns.name = self.border_identifier
        result.index.name = line_flow_data.sent_up.index.name
        return result
