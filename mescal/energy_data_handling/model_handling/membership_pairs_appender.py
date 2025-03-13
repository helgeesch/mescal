from abc import ABC, abstractmethod
from typing import Literal
import pandas as pd

from mescal.utils.pandas_utils.set_new_column import set_column
from mescal.utils.multi_key_utils.common_base_key_finder import CommonBaseKeyFinder


class BaseMembershipPairsAppender(ABC):
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
        return self._append_combo_columns(df_with_from_to_columns, 'directional')

    def append_sorted_combo_columns(
            self,
            df_with_from_to_columns: pd.DataFrame,
    ) -> pd.DataFrame:
        return self._append_combo_columns(df_with_from_to_columns, 'sorted')

    def append_opposite_combo_columns(
            self,
            df_with_from_to_columns: pd.DataFrame,
    ) -> pd.DataFrame:
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
        return a.astype(str) + self._separator + b.astype(str)

    def _combine_sorted_values(self, a: pd.Series, b: pd.Series) -> pd.Series:
        return pd.DataFrame({
            'a': a.astype(str),
            'b': b.astype(str)
        }).apply(lambda x: self._separator.join(sorted([x['a'], x['b']])), axis=1)


class TupleMembershipPairsAppender(BaseMembershipPairsAppender):
    def _combine_values(self, a: pd.Series, b: pd.Series) -> pd.Series:
        return pd.Series([tuple(x) for x in zip(a, b)], index=a.index)

    def _combine_sorted_values(self, a: pd.Series, b: pd.Series) -> pd.Series:
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
