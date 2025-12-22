from abc import ABC, abstractmethod
from typing import Literal
import pandas as pd

from mesqual.utils.pandas_utils.set_new_column import set_column
from mesqual.utils.multi_key_utils.common_base_key_finder import CommonBaseKeyFinder


class BaseMembershipPairsAppender(ABC):
    """Abstract base class for creating combination identifiers from paired energy system relationships.
    
    In energy system modeling, many entities have directional relationships that require
    unique identification for analysis. This class provides a unified framework for creating
    combination identifiers from paired columns, particularly useful for:
    
    - Transmission lines connecting nodes (node_from/node_to combinations)
    - Regional trade flows (region_from/region_to pairs)
    - Pipeline connections (hub_from/hub_to relationships)
    - Market interconnections (market_from/market_to links)
    
    The class supports three distinct combination strategies:
    
    1. **Directional**: Preserves relationship direction (A→B ≠ B→A)
        - Essential for analyzing flow directions, capacity constraints, and directional costs
    
    2. **Sorted**: Creates bidirectional identifiers (A→B = B→A becomes A-B)
        - Useful for identifying unique connections regardless of direction
    
    3. **Opposite**: Reverses relationship direction (A→B becomes B→A)
        - Enables reverse flow analysis and bidirectional modeling
    
    This abstraction enables different implementation strategies (string concatenation,
    tuple creation, etc.) while maintaining consistent naming patterns across MESQUAL
    energy data models.
    
    Args:
        from_identifier: Suffix/prefix identifying source/origin columns. Defaults to '_from'.
        to_identifier: Suffix/prefix identifying destination/target columns. Defaults to '_to'.
        combo_col_suffix: Suffix for directional combination column names. Defaults to '_combo'.
        combo_col_prefix: Prefix for directional combination column names. Defaults to None.
        sorted_combo_col_suffix: Suffix for sorted combination column names. Defaults to '_combo_sorted'.
        sorted_combo_col_prefix: Prefix for sorted combination column names. Defaults to None.
        opposite_combo_col_suffix: Suffix for opposite combination column names. Defaults to '_combo_opposite'.
        opposite_combo_col_prefix: Prefix for opposite combination column names. Defaults to None.
        
    Raises:
        ValueError: If neither suffix nor prefix is provided for any combination type.
        
    Note:
        Either suffix or prefix must be specified for each combination type to ensure
        proper column naming conventions.
        
    Examples:

        >>> # For transmission line analysis
        >>> appender = StringMembershipPairsAppender(separator=' → ')
        >>> lines_df = appender.append_combo_columns(transmission_df)
        >>> # Creates 'node_combo' column: 'NodeA → NodeB'
        
        >>> # For bidirectional connections
        >>> lines_df = appender.append_sorted_combo_columns(transmission_df)
        >>> # Creates 'node_combo_sorted' column: 'NodeA → NodeB' (alphabetical)
        >>> lines_df = appender.append_opposite_combo_columns(transmission_df)
        >>> # Creates 'node_combo_opposite' column: 'NodeB → NodeA' (alphabetical)
    """
    def __init__(
            self,
            from_identifier: str = '_from',
            to_identifier: str = '_to',
            combo_col_suffix: str = '_combo',
            combo_col_prefix: str = None,
            sorted_combo_col_suffix: str = '_combo_sorted',
            sorted_combo_col_prefix: str = None,
            opposite_combo_col_suffix: str = '_combo_opposite',
            opposite_combo_col_prefix: str = None,
    ):
        self._from_identifier = from_identifier
        self._to_identifier = to_identifier

        self._combo_col_suffix = combo_col_suffix or ''
        self._combo_col_prefix = combo_col_prefix or ''
        if not any([self._combo_col_suffix, self._combo_col_prefix]):
            raise ValueError("Must specify either suffix or prefix for combo columns")

        self._sorted_combo_col_suffix = sorted_combo_col_suffix or ''
        self._sorted_combo_col_prefix = sorted_combo_col_prefix or ''
        if not any([self._sorted_combo_col_suffix, self._sorted_combo_col_prefix]):
            raise ValueError("Must specify either suffix or prefix for sorted combo columns")

        self._opposite_combo_col_suffix = opposite_combo_col_suffix or ''
        self._opposite_combo_col_prefix = opposite_combo_col_prefix or ''
        if not any([self._opposite_combo_col_suffix, self._opposite_combo_col_prefix]):
            raise ValueError("Must specify either suffix or prefix for opposite combo columns")

        self._common_base_key_finder = CommonBaseKeyFinder(from_identifier, to_identifier)

    def append_combo_columns(
            self,
            df_with_from_to_columns: pd.DataFrame,
    ) -> pd.DataFrame:
        """Creates directional combination columns preserving relationship direction.
        
        Generates combination identifiers that maintain the original direction of
        relationships, essential for energy system analysis where flow direction,
        capacity constraints, and directional costs matter.
        
        This method automatically identifies all paired columns (those with matching
        base names plus from/to identifiers) and creates directional combinations
        using the configured naming strategy.
        
        Args:
            df_with_from_to_columns: DataFrame containing paired relationship columns
                                   (e.g., 'node_from'/'node_to', 'region_from'/'region_to')
        
        Returns:
            Enhanced DataFrame with directional combination columns added.
            Original data preserved, new columns follow configured naming pattern.
            
        Examples:
            Transmission line directional analysis:
            
            >>> # DataFrame with node connections
            >>> lines_df = pd.DataFrame({
            ...     'line_id': ['L1', 'L2', 'L3'],
            ...     'node_from': ['NodeA', 'NodeB', 'NodeC'],
            ...     'node_to': ['NodeB', 'NodeA', 'NodeA'],
            ...     'capacity_mw': [1000, 800, 600]
            ... })
            >>> 
            >>> appender = StringMembershipPairsAppender(separator=' → ')
            >>> result = appender.append_combo_columns(lines_df)
            >>> # Result includes 'node_combo': ['NodeA → NodeB', 'NodeB → NodeA', 'NodeC → NodeA']
        """
        return self._append_combo_columns(df_with_from_to_columns, 'directional')

    def append_sorted_combo_columns(
            self,
            df_with_from_to_columns: pd.DataFrame,
    ) -> pd.DataFrame:
        """Creates bidirectional combination columns with alphabetical ordering.
        
        Generates combination identifiers that treat relationships as bidirectional
        by sorting the paired values alphabetically. This is particularly useful for
        identifying unique connections regardless of direction, such as:
        
        - Transmission line corridors (same physical connection)
        - Regional trade partnerships (bidirectional trade agreements)
        - Pipeline systems (flow can be reversed)
        - Market coupling arrangements (mutual price influence)
        
        Args:
            df_with_from_to_columns: DataFrame containing paired relationship columns
                                   with potential bidirectional connections
        
        Returns:
            Enhanced DataFrame with sorted combination columns added.
            Bidirectional relationships receive identical identifiers.
            
        Examples:
            Bidirectional connection analysis:
            
            >>> # DataFrame with potentially bidirectional connections
            >>> connections_df = pd.DataFrame({
            ...     'connection_id': ['C1', 'C2', 'C3'],
            ...     'region_from': ['DE', 'FR', 'DE'],
            ...     'region_to': ['FR', 'DE', 'NL'],
            ...     'trade_capacity': [2000, 2000, 1500]
            ... })
            >>> 
            >>> appender = StringMembershipPairsAppender(separator='-')
            >>> result = appender.append_sorted_combo_columns(connections_df)
            >>> # Result includes 'region_combo_sorted': ['DE-FR', 'DE-FR', 'DE-NL']
            >>> # Note: 'DE→FR' and 'FR→DE' both become 'DE-FR'
        """
        return self._append_combo_columns(df_with_from_to_columns, 'sorted')

    def append_opposite_combo_columns(
            self,
            df_with_from_to_columns: pd.DataFrame,
    ) -> pd.DataFrame:
        """Creates reverse-direction combination columns for opposite flow analysis.
        
        Generates combination identifiers with reversed direction, enabling analysis
        of reverse flows, return paths, and bidirectional modeling scenarios. This
        is particularly valuable for:
        
        - Reverse power flows in transmission networks
        - Return commodity flows in pipeline systems
        - Opposite direction trade flows
        - Backup routing analysis
        
        The method swaps the from/to values before creating combinations, effectively
        creating the opposite directional identifier for each relationship.
        
        Args:
            df_with_from_to_columns: DataFrame containing directional relationships
                                   where reverse analysis is needed
        
        Returns:
            Enhanced DataFrame with opposite-direction combination columns added.
            Each relationship receives its reverse-direction identifier.
            
        Examples:
            Reverse flow analysis:
            
            >>> # DataFrame with primary flow directions
            >>> flows_df = pd.DataFrame({
            ...     'flow_id': ['F1', 'F2', 'F3'],
            ...     'hub_from': ['HubA', 'HubB', 'HubC'],
            ...     'hub_to': ['HubB', 'HubC', 'HubA'],
            ...     'primary_flow': [100, 150, 80]
            ... })
            >>> 
            >>> appender = StringMembershipPairsAppender(separator=' ← ')
            >>> result = appender.append_opposite_combo_columns(flows_df)
            >>> # Result includes 'hub_combo_opposite': ['HubB ← HubA', 'HubC ← HubB', 'HubA ← HubC']
            >>> # Useful for modeling reverse flow scenarios
        """
        return self._append_combo_columns(df_with_from_to_columns, 'opposite')

    def _append_combo_columns(
            self,
            df_with_from_to_columns: pd.DataFrame,
            which_combo: Literal['directional', 'sorted', 'opposite']
    ) -> pd.DataFrame:
        from_id = self._from_identifier
        to_id = self._to_identifier

        _col_names = df_with_from_to_columns.columns
        base_columns = self._common_base_key_finder.get_keys_for_which_all_association_tags_appear(_col_names)

        for base in base_columns:
            from_col = f'{base}{from_id}'
            to_col = f'{base}{to_id}'

            _from_values = df_with_from_to_columns[from_col]
            _to_values = df_with_from_to_columns[to_col]

            if which_combo == 'directional':
                new_col = f'{self._combo_col_prefix}{base}{self._combo_col_suffix}'
                new_values = self._combine_values(_from_values, _to_values)
            elif which_combo == 'sorted':
                new_col = f'{self._sorted_combo_col_prefix}{base}{self._sorted_combo_col_suffix}'
                new_values = self._combine_sorted_values(_from_values, _to_values)
            elif which_combo == 'opposite':
                new_col = f'{self._opposite_combo_col_prefix}{base}{self._opposite_combo_col_suffix}'
                new_values = self._combine_values(_to_values, _from_values)
            else:
                raise NotImplementedError

            df_with_from_to_columns = set_column(df_with_from_to_columns, new_col, new_values)

        return df_with_from_to_columns

    @abstractmethod
    def _combine_values(self, a: pd.Series, b: pd.Series) -> pd.Series:
        pass

    @abstractmethod
    def _combine_sorted_values(self, a: pd.Series, b: pd.Series) -> pd.Series:
        pass


class StringMembershipPairsAppender(BaseMembershipPairsAppender):
    """String-based implementation for creating combination identifiers from energy system relationships.
    
    This concrete implementation creates human-readable string combinations from paired
    relationships using configurable separators. Particularly well-suited for:
    
    - Data visualization and reporting (readable connection labels)
    - User interface displays (network connection names)
    - Export formats requiring string identifiers
    - Debugging and data exploration
    
    The class inherits all combination strategies from the base class while implementing
    string-specific combination logic with customizable separators for different use cases.
    
    Args:
        separator: String used to join paired values in combinations. Defaults to ' - '.
                  Common patterns: ' → ' (directional), '-' (neutral), ' <-> ' (bidirectional)
        **kwargs: All arguments from BaseMembershipPairsAppender for column naming configuration
    
    Examples:
        Energy system string combinations:
        
        >>> # Transmission line connections with directional separator
        >>> appender = StringMembershipPairsAppender(separator=' → ')
        >>> lines_with_combos = appender.append_combo_columns(transmission_df)
        >>> # Creates readable labels: 'NodeA → NodeB', 'NodeB → NodeC'
        
        >>> # Regional trade connections with neutral separator
        >>> trade_appender = StringMembershipPairsAppender(separator='-')
        >>> trade_with_combos = trade_appender.append_sorted_combo_columns(trade_df)
        >>> # Creates trade corridor labels: 'DE-FR', 'FR-NL'
    """
    def __init__(
            self,
            from_identifier: str = '_from',
            to_identifier: str = '_to',
            combo_col_suffix: str = '_combo',
            combo_col_prefix: str = None,
            sorted_combo_col_suffix: str = '_combo_sorted',
            sorted_combo_col_prefix: str = None,
            opposite_combo_col_suffix: str = '_combo_opposite',
            opposite_combo_col_prefix: str = None,
            separator: str = ' - ',
    ):
        """Initialize the string-based membership pairs appender.
        
        Args:
            from_identifier: Suffix/prefix identifying source/origin columns. Defaults to '_from'.
            to_identifier: Suffix/prefix identifying destination/target columns. Defaults to '_to'.
            combo_col_suffix: Suffix for directional combination column names. Defaults to '_combo'.
            combo_col_prefix: Prefix for directional combination column names. Defaults to None.
            sorted_combo_col_suffix: Suffix for sorted combination column names. Defaults to '_combo_sorted'.
            sorted_combo_col_prefix: Prefix for sorted combination column names. Defaults to None.
            opposite_combo_col_suffix: Suffix for opposite combination column names. Defaults to '_combo_opposite'.
            opposite_combo_col_prefix: Prefix for opposite combination column names. Defaults to None.
            separator: String separator for joining paired values. Defaults to ' - '.
                      Use ' → ' for directional flows, '-' for neutral connections,
                      ' <-> ' for bidirectional relationships.
        """
        super().__init__(
            from_identifier=from_identifier,
            to_identifier=to_identifier,
            combo_col_suffix=combo_col_suffix,
            combo_col_prefix=combo_col_prefix,
            sorted_combo_col_suffix=sorted_combo_col_suffix,
            sorted_combo_col_prefix=sorted_combo_col_prefix,
            opposite_combo_col_suffix=opposite_combo_col_suffix,
            opposite_combo_col_prefix=opposite_combo_col_prefix,
        )
        self._separator = separator

    def _combine_values(self, a: pd.Series, b: pd.Series) -> pd.Series:
        """Combines two series into directional string combinations.
        
        Args:
            a: Source/origin values (from column)
            b: Destination/target values (to column)
            
        Returns:
            Series with string combinations preserving a→b direction
        """
        return a.astype(str) + self._separator + b.astype(str)

    def _combine_sorted_values(self, a: pd.Series, b: pd.Series) -> pd.Series:
        """Combines two series into bidirectional string combinations with alphabetical ordering.
        
        Args:
            a: First set of relationship values
            b: Second set of relationship values
            
        Returns:
            Series with alphabetically sorted string combinations (bidirectional)
        """
        if a.empty and b.empty:
            return pd.Series()
        return pd.DataFrame({
            'a': a.astype(str),
            'b': b.astype(str)
        }).apply(lambda x: self._separator.join(sorted([x['a'], x['b']])), axis=1)


class TupleMembershipPairsAppender(BaseMembershipPairsAppender):
    """Tuple-based implementation for creating combination identifiers from energy system relationships.
    
    This concrete implementation creates tuple combinations from paired relationships,
    offering several advantages for programmatic use:
    
    - Memory efficiency (no string concatenation overhead)
    - Fast equality comparisons and set operations
    - Preservation of original data types
    - Direct use as dictionary keys or index values
    - Integration with pandas MultiIndex structures
    
    Particularly valuable for:
    
    - High-performance energy system simulations
    - Large-scale network analysis
    - Optimization model formulations
    - Internal data processing pipelines
    
    The tuple format maintains the exact relationship structure while enabling
    efficient programmatic manipulation of connection data.
    
    Examples:
        Energy system tuple combinations:
        
        >>> # Transmission network with tuple identifiers
        >>> appender = TupleMembershipPairsAppender()
        >>> lines_with_tuples = appender.append_combo_columns(transmission_df)
        >>> # Creates tuple identifiers: ('NodeA', 'NodeB'), ('NodeB', 'NodeC')
        
        >>> # Bidirectional connections for optimization
        >>> lines_bidirectional = appender.append_sorted_combo_columns(transmission_df)
        >>> # Creates sorted tuples: ('NodeA', 'NodeB'), ('NodeB', 'NodeC')
        >>> # Useful as keys in optimization constraints
    """
    def _combine_values(self, a: pd.Series, b: pd.Series) -> pd.Series:
        """Combines two series into directional tuple combinations.
        
        Args:
            a: Source/origin values (from column)
            b: Destination/target values (to column)
            
        Returns:
            Series with tuple combinations preserving (a, b) direction
        """
        return pd.Series([tuple(x) for x in zip(a, b)], index=a.index)

    def _combine_sorted_values(self, a: pd.Series, b: pd.Series) -> pd.Series:
        """Combines two series into bidirectional tuple combinations with alphabetical ordering.
        
        Args:
            a: First set of relationship values
            b: Second set of relationship values
            
        Returns:
            Series with alphabetically sorted tuple combinations (bidirectional)
        """
        return pd.Series([tuple(sorted([x, y])) for x, y in zip(a, b)], index=a.index)


if __name__ == '__main__':
    data = {
        'node_from': ['A', 'B', 'C'],
        'node_to': ['B', 'A', 'A'],
        'zone_from': ['DE', 'FR', 'NL'],
        'zone_to': ['FR', 'DE', 'DE']
    }
    df = pd.DataFrame(data)

    string_appender = StringMembershipPairsAppender(separator=' <-> ')
    result_str = string_appender.append_combo_columns(df)
    result_str = string_appender.append_sorted_combo_columns(result_str)
    result_str = string_appender.append_opposite_combo_columns(result_str)
    print("\nString format:")
    print(result_str)

    tuple_appender = TupleMembershipPairsAppender()
    result_tuple = tuple_appender.append_combo_columns(df)
    result_tuple = tuple_appender.append_sorted_combo_columns(result_tuple)
    result_tuple = tuple_appender.append_opposite_combo_columns(result_tuple)
    print("\nTuple format:")
    print(result_tuple)
