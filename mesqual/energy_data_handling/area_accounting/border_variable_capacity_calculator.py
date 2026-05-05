from typing import Literal

import pandas as pd

from mesqual.energy_data_handling.network_lines_data import NetworkLineCapacitiesData
from mesqual.energy_data_handling.area_accounting.border_variable_base import AreaBorderVariableCalculatorBase


class BorderCapacityCalculator(AreaBorderVariableCalculatorBase):
    """Calculates aggregated transmission capacities for area borders.
    
    This calculator aggregates line-level transmission capacities to border level,
    handling bidirectional capacity data with proper directional aggregation.
    
    Example:

        >>> from mesqual.energy_data_handling.network_lines_data import NetworkLineCapacitiesData
        >>> import pandas as pd
        >>> 
        >>> # Create capacity data
        >>> time_index = pd.date_range('2024-01-01', periods=24, freq='h')
        >>> capacities = NetworkLineCapacitiesData(
        ...     capacities_up=pd.DataFrame({...}),
        ...     capacities_down=pd.DataFrame({...})
        ... )
        >>> 
        >>> calculator = BorderCapacityCalculator(
        ...     area_border_model_df, line_model_df, node_model_df, 'country'
        ... )
        >>> 
        >>> # Calculate capacities for up direction (area_from → area_to)
        >>> up_capacities = calculator.calculate(capacities, direction='up')
        >>> print(up_capacities)
    """

    @property
    def variable_name(self) -> str:
        return "border_capacity"

    def calculate(
            self,
            line_capacity_data: NetworkLineCapacitiesData,
            direction: Literal['up', 'down'] = 'up'
    ) -> pd.DataFrame:
        """Aggregate line-level transmission capacities to border level.
        
        Sums transmission capacities of all lines belonging to each border,
        respecting the specified direction and handling bidirectional capacity data.
        Lines are aggregated based on their topological relationship to the border.
        
        Direction logic:
        - 'up': Capacities for flows from area_from to area_to
        - 'down': Capacities for flows from area_to to area_from
        - 'total': Sum capacities in both directions

        For each border, the method:
        1. Identifies lines in 'up' and 'down' topological directions
        2. Selects appropriate capacity data based on requested direction
        3. Sums capacities across all border lines
        4. Handles missing data by excluding unavailable lines
        
        Args:
            line_capacity_data: NetworkLineCapacitiesData containing bidirectional
                capacity time series. Must include capacities_up and capacities_down
                DataFrames with line IDs as columns and timestamps as index.
            direction: Direction for capacity aggregation:
                - 'up': Sum capacities for area_from → area_to flows
                - 'down': Sum capacities for area_to → area_from flows
                
        Returns:
            DataFrame with border-level capacity aggregations. Index matches the 
            input capacity data, columns are border identifiers. Values represent
            total transmission capacity in MW for each border and timestamp.
            
        Raises:
            ValueError: If direction is not 'up', 'down', or 'total'
            
        Example:
            
            >>> # Calculate up-direction capacities (exports from area_from)
            >>> up_caps = calculator.calculate(capacity_data, direction='up')
            >>> 
            >>> # Calculate down-direction capacities (imports to area_from)  
            >>> down_caps = calculator.calculate(capacity_data, direction='down')
            >>> 
            >>> print(f"DE→FR capacity: {up_caps.loc['2024-01-01 12:00', 'DE-FR']:.0f} MW")
        """
        if direction == 'total':
            return self.calculate(line_capacity_data, 'up') + self.calculate(line_capacity_data, 'down')

        self._validate_time_series_data(line_capacity_data.capacities_up, "capacities_up")
        self._validate_time_series_data(line_capacity_data.capacities_down, "capacities_down")
        
        border_capacities = {}

        for border_id, border in self.area_border_model_df.iterrows():
            lines_up, lines_down = self.get_border_lines_in_topological_up_and_down_direction(border_id)

            if not lines_up and not lines_down:
                # No lines found for this border - create empty series
                index = line_capacity_data.capacities_up.index
                border_capacities[border_id] = pd.Series(index=index, dtype=float)
                continue

            if direction == 'up':
                # For 'up' direction: use up capacities of lines_up + down capacities of lines_down
                capacity_parts = []
                if lines_up:
                    available_lines_up = [line for line in lines_up if line in line_capacity_data.capacities_up.columns]
                    if available_lines_up:
                        capacity_parts.append(line_capacity_data.capacities_up[available_lines_up])
                        
                if lines_down:
                    available_lines_down = [line for line in lines_down if line in line_capacity_data.capacities_down.columns]
                    if available_lines_down:
                        capacity_parts.append(line_capacity_data.capacities_down[available_lines_down])
                        
            elif direction == 'down':
                # For 'down' direction: use down capacities of lines_up + up capacities of lines_down
                capacity_parts = []
                if lines_up:
                    available_lines_up = [line for line in lines_up if line in line_capacity_data.capacities_down.columns]
                    if available_lines_up:
                        capacity_parts.append(line_capacity_data.capacities_down[available_lines_up])
                        
                if lines_down:
                    available_lines_down = [line for line in lines_down if line in line_capacity_data.capacities_up.columns]
                    if available_lines_down:
                        capacity_parts.append(line_capacity_data.capacities_up[available_lines_down])
            else:
                raise ValueError(f"Unknown capacity direction: {direction}. Must be 'up' or 'down'")

            # Combine and sum capacities
            if capacity_parts:
                all_capacities = pd.concat(capacity_parts, axis=1)
                border_capacities[border_id] = all_capacities.sum(axis=1)
            else:
                # No capacity data available for any lines
                index = line_capacity_data.capacities_up.index
                border_capacities[border_id] = pd.Series(index=index, dtype=float)

        result = pd.DataFrame(border_capacities)
        result.columns.name = self.border_identifier
        return result


if __name__ == "__main__":
    import numpy as np

    # Create example network structure
    node_model_df = pd.DataFrame({
        'country': ['DE', 'DE', 'FR', 'FR', 'BE'],
        'voltage': [380, 220, 380, 220, 380]
    }, index=['DE1', 'DE2', 'FR1', 'FR2', 'BE1'])
    
    line_model_df = pd.DataFrame({
        'node_from': ['DE1', 'DE2', 'FR1'],  
        'node_to': ['FR1', 'FR2', 'BE1'],
        'capacity': [1000, 800, 1200]
    }, index=['DE-FR_Line1', 'DE-FR_Line2', 'FR-BE_Line1'])
    
    area_border_model_df = pd.DataFrame(index=['DE - FR', 'FR - DE', 'FR - BE'])
    
    print("Network structure:")
    print("Node model:")
    print(node_model_df)
    print("\nLine model:")  
    print(line_model_df)
    print("\nBorder model:")
    print(area_border_model_df)
    
    # Create time series capacity data
    time_index = pd.date_range('2024-01-01', periods=168, freq='h')  # 1 week
    n_lines = len(line_model_df)
    
    # Simulate variable capacities (e.g., due to temperature, maintenance)
    base_capacities = np.array([1000, 800, 1200])  # Base capacity for each line
    
    # Add some realistic variation
    capacities_up_data = np.random.normal(
        base_capacities, 
        base_capacities * 0.05,  # 5% standard deviation
        size=(len(time_index), n_lines)
    )
    capacities_up_data = np.clip(capacities_up_data, base_capacities * 0.8, base_capacities * 1.0)
    
    # Assume asymmetric capacities (down direction slightly different)
    capacities_down_data = capacities_up_data * np.random.uniform(0.9, 1.1, size=capacities_up_data.shape)
    
    capacity_data = NetworkLineCapacitiesData(
        capacities_up=pd.DataFrame(
            capacities_up_data,
            index=time_index,
            columns=line_model_df.index
        ),
        capacities_down=pd.DataFrame(
            capacities_down_data, 
            index=time_index,
            columns=line_model_df.index
        )
    )
    
    print(f"\nLine capacity data (first 5 hours):")
    print("Up direction:")
    print(capacity_data.capacities_up.head())
    print("\nDown direction:")
    print(capacity_data.capacities_down.head())
    
    # Initialize calculator
    calculator = BorderCapacityCalculator(
        area_border_model_df=area_border_model_df,
        line_model_df=line_model_df,
        node_model_df=node_model_df,
        area_column='country'
    )
    
    # Test line direction identification
    print(f"\nBorder line analysis:")
    for border_id in area_border_model_df.index:
        lines_up, lines_down = calculator.get_border_lines_in_topological_up_and_down_direction(border_id)
        area_from, area_to = calculator.decompose_area_border_name_to_areas(border_id) 
        print(f"{border_id} ({area_from} → {area_to}):")
        print(f"  Lines up (from {area_from} to {area_to}): {lines_up}")
        print(f"  Lines down (from {area_to} to {area_from}): {lines_down}")
    
    # Calculate border capacities for both directions
    up_capacities = calculator.calculate(capacity_data, direction='up')
    down_capacities = calculator.calculate(capacity_data, direction='down')
    
    print(f"\nBorder capacity results:")
    print("Up direction capacities (first 5 hours):")
    print(up_capacities.head())
    print("\nDown direction capacities (first 5 hours):")
    print(down_capacities.head())
    
    # Show some summary statistics
    print(f"\nCapacity statistics (MW):")
    print("Up direction:")
    for border in up_capacities.columns:
        mean_cap = up_capacities[border].mean()
        min_cap = up_capacities[border].min()
        max_cap = up_capacities[border].max()
        print(f"  {border}: Mean={mean_cap:.0f}, Min={min_cap:.0f}, Max={max_cap:.0f}")
        
    print("Down direction:")
    for border in down_capacities.columns:
        mean_cap = down_capacities[border].mean()
        min_cap = down_capacities[border].min() 
        max_cap = down_capacities[border].max()
        print(f"  {border}: Mean={mean_cap:.0f}, Min={min_cap:.0f}, Max={max_cap:.0f}")
