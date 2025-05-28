import pandas as pd


def sort_multiindex(df: pd.DataFrame, custom_order: list[str | int], level: str | int, axis: int = 0) -> pd.DataFrame:
    """
    Sort a DataFrame's MultiIndex at a specific level using a custom order.

    Reorders the specified level according to custom_order while preserving
    the existing order of all other levels. This allows for sequential sorting
    operations where each sort maintains previous orderings.

    Values in the target level that are not included in custom_order will be
    appended at the end, maintaining their original relative order.

    Examples
    --------
    >>> idx = pd.MultiIndex.from_arrays([['A', 'A', 'B', 'B'], [1, 2, 1, 2]])
    >>> df = pd.DataFrame({'val': [10, 20, 30, 40]}, index=idx)
    >>> sort_multiindex(df, [2, 1], level=1)  # Sort second level
    """
    if axis not in [0, 1]:
        raise ValueError("axis must be 0 (rows) or 1 (columns)")

    if isinstance(level, str):
        level_num = df.axes[axis].names.index(level)
    elif isinstance(level, int):
        level_num = level
    else:
        raise ValueError("level must be a string (level name) or an integer (level number)")

    idx = df.axes[axis]
    idx_tuples = idx.to_list()
    remaining_values_in_level_to_sort = [i for i in idx.get_level_values(level_num).unique() if i not in custom_order]
    ordered_tuples = []
    for value in custom_order + remaining_values_in_level_to_sort:
        for tuple_item in idx_tuples:
            if tuple_item[level_num] == value:
                ordered_tuples.append(tuple_item)

    new_index = pd.MultiIndex.from_tuples(ordered_tuples, names=idx.names)

    if axis == 0:
        return df.reindex(new_index)
    else:
        return df.reindex(columns=new_index)


if __name__ == '__main__':
    arrays = [
        ['B', 'B', 'B', 'A', 'A', 'A', 'C', 'C', 'C'],
        ['x', 'y', 'z', 'x', 'y', 'z', 'x', 'y', 'z'],
        ['1', '2', '3', '3', '1', '2', '2', '3', '1']
    ]
    index = pd.MultiIndex.from_arrays(arrays, names=('letter', 'symbol', 'number'))
    df = pd.DataFrame({'val': range(9)}, index=index)

    print("Original DataFrame:")
    print(df)
    print("\nAfter sorting 'symbol' level with order ['z', 'x', 'y']:")
    df_sorted = sort_multiindex(df, ['z', 'x', 'y'], 'symbol')
    print(df_sorted)
    print("\nNow sorting 'number' level with order ['3', '1', '2']:")
    df_sorted2 = sort_multiindex(df_sorted, ['3', '1', '2'], 'number')
    print(df_sorted2)