from typing import Union, List, Literal, Callable, Dict, Any
from datetime import time
import copy
import calendar
import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

X_AXIS_AGGS = Literal['date', 'year_month', 'year_week', 'week', 'month', 'year']
X_AXIS_TYPES = Union[X_AXIS_AGGS, List[X_AXIS_AGGS]]
GROUPBY_AGG_TYPES = Union[str, List[str]]


class DashboardConfig:
    DEFAULT_STATISTICS = {
        'Datums': lambda x: len(x),
        'Abs max': lambda x: x.abs().max(),
        'Abs mean': lambda x: x.abs().mean(),
        'Max': lambda x: x.max(),
        'Mean': lambda x: x.mean(),
        'Min': lambda x: x.min(),
    }

    STATISTICS_LIBRARY = {
        '# Values': lambda x: (~x.isna()).sum(),
        '# NaNs': lambda x: x.isna().sum(),
        '% == 0': lambda x: (x.round(2) == 0).sum() / (~x.isna()).sum() * 100,
        '% != 0': lambda x: ((x.round(2) != 0) & (~x.isna())).sum() / (~x.isna()).sum() * 100,
        '% > 0': lambda x: (x.round(2) > 0).sum() / (~x.isna()).sum() * 100,
        '% < 0': lambda x: (x.round(2) < 0).sum() / (~x.isna()).sum() * 100,
        'Mean of v>0': lambda x: x.where(x > 0, np.nan).mean(),
        'Mean of v<0': lambda x: x.where(x < 0, np.nan).mean(),
        'Median': lambda x: x.median(),
        'Q0.99': lambda x: x.quantile(0.99),
        'Q0.95': lambda x: x.quantile(0.95),
        'Q0.05': lambda x: x.quantile(0.05),
        'Q0.01': lambda x: x.quantile(0.01),
        'Std': lambda x: x.std(),
    }

    def __init__(
            self,
            x_axis: X_AXIS_TYPES = 'date',
            facet_col: str = None,
            facet_row: str = None,
            facet_col_wrap: int = None,
            facet_col_order: list[str] = None,
            facet_row_order: list[str] = None,
            ratio_of_stat_col: float = 0.1,
            stat_aggs: Dict[str, Callable[[pd.Series], float | int]] = None,
            groupby_aggregation: GROUPBY_AGG_TYPES = 'mean',
            title: str = None,
            color_continuous_scale: str | list[str] | list[tuple[float, str]] = 'Turbo',
            color_continuous_midpoint: int | float = None,
            range_color: list[int | float] = None,
            per_facet_col_colorscale: bool = False,
            per_facet_row_colorscale: bool = False,
            facet_row_color_settings: dict = None,
            facet_col_color_settings: dict = None,
            subplots_vertical_spacing: float = None,
            subplots_horizontal_spacing: float = None,
            time_series_figure_kwargs: dict = None,
            stat_figure_kwargs: dict = None,
            universal_figure_kwargs: dict = None,
            **figure_kwargs
    ):
        self.x_axis = x_axis
        self.facet_col = facet_col
        self.facet_row = facet_row
        self.facet_col_wrap = facet_col_wrap
        self.facet_col_order = facet_col_order
        self.facet_row_order = facet_row_order
        self.ratio_of_stat_col = ratio_of_stat_col
        self.stat_aggs = stat_aggs or self.DEFAULT_STATISTICS
        self.groupby_aggregation = groupby_aggregation
        self.title = title

        self.per_facet_col_colorscale = per_facet_col_colorscale
        self.per_facet_row_colorscale = per_facet_row_colorscale

        if per_facet_col_colorscale and per_facet_row_colorscale:
            raise ValueError("Cannot use both per_facet_col_colorscale and per_facet_row_colorscale simultaneously")
        if facet_row_color_settings and not per_facet_row_colorscale:
            raise ValueError("Set per_facet_row_colorscale to True in order to use facet_row_color_settings.")
        if facet_col_color_settings and not per_facet_col_colorscale:
            raise ValueError("Set per_facet_col_colorscale to True in order to use facet_col_color_settings.")

        self.facet_row_color_settings = facet_row_color_settings or {}
        self.facet_col_color_settings = facet_col_color_settings or {}

        self.time_series_figure_kwargs = time_series_figure_kwargs or {}
        self.stat_figure_kwargs = stat_figure_kwargs or {}

        self.subplots_vertical_spacing = subplots_vertical_spacing
        self.subplots_horizontal_spacing = subplots_horizontal_spacing

        universal_figure_kwargs = universal_figure_kwargs or {}

        self.figure_kwargs = {
            'color_continuous_scale': color_continuous_scale,
            'color_continuous_midpoint': color_continuous_midpoint,
            'range_color': range_color,
            **universal_figure_kwargs,
            **figure_kwargs,
        }


class DataProcessor:
    @staticmethod
    def validate_input_data_and_config(data: pd.DataFrame, config: DashboardConfig) -> None:
        x_axis = config.x_axis
        groupby_aggregation = config.groupby_aggregation
        facet_col = config.facet_col
        facet_row = config.facet_row
        facet_col_wrap = config.facet_col_wrap

        if facet_col_wrap is not None and facet_row is not None:
            raise ValueError('You cannot set facet_row if you are setting a facet_col_wrap')

        if isinstance(data, pd.Series):
            if sum(facet not in [None, 'x_axis', 'groupby_aggregation'] for facet in [facet_col, facet_row]):
                raise ValueError('You can not define facet_col or facet_row if you just have a pd.Series')
        elif data.columns.nlevels > 2:
            raise ValueError('Your data must not have more than 2 column index levels.')
        elif data.columns.nlevels == 2:
            if (facet_col is None) and (facet_row is None):
                raise ValueError('If you have two column levels, you must define both, facet_col and facet_row.')
            if isinstance(x_axis, list) or isinstance(groupby_aggregation, list):
                raise ValueError(
                    'You cannot set x_axis or groupby_aggregation to a list if your data already has 2 levels.'
                )
        elif data.columns.nlevels == 1:
            if sum(facet not in [None, 'x_axis', 'groupby_aggregation'] for facet in [facet_col, facet_row]) > 1:
                raise ValueError('You only have 1 column level. You can only define facet_col or facet_row')
            if isinstance(x_axis, list) and isinstance(groupby_aggregation, list):
                raise ValueError(
                    'You cannot set x_axis and groupby_aggregation to a list if your data already has 1 level.'
                )

        if isinstance(x_axis, list):
            if not any('x_axis' == facet for facet in [facet_col, facet_row]):
                raise ValueError(
                    "x_axis must be either 'facet_col' or 'facet_row' when provided as a list."
                )
        else:
            if any('x_axis' == facet for facet in [facet_col, facet_row]):
                raise ValueError(
                    "You provided a str for x_axis, "
                    "but set facet_col or facet_row to 'x_axis'. This is not allowed! \n"
                    "You must provide a List[str] and in order to use facet_row / facet_col "
                    "for different x_axis."
                )

        if isinstance(groupby_aggregation, list):
            if not any('groupby_aggregation' == facet for facet in [facet_col, facet_row]):
                raise ValueError(
                    "groupby_aggregation must be either 'facet_col' or 'facet_row' when provided as a list."
                )
        else:
            if any('groupby_aggregation' == facet for facet in [facet_col, facet_row]):
                raise ValueError(
                    "You provided a str for groupby_aggregation, "
                    "but set facet_col or facet_row to 'groupby_aggregation'. This is not allowed! \n"
                    "You must provide a List[str] and in order to use facet_row / facet_col "
                    "for different groupby_aggregation."
                )

    @staticmethod
    def prepare_dataframe_for_facet(data: pd.DataFrame, config: DashboardConfig) -> pd.DataFrame:
        for k in ['x_axis', 'groupby_aggregation']:
            config_value = getattr(config, k)
            if isinstance(config_value, list):
                data = pd.concat(
                    {i: data.copy(deep=True) for i in config_value},
                    axis=1,
                    names=[k],
                )
        return data

    @staticmethod
    def ensure_df_has_two_column_levels(data: pd.DataFrame, config: DashboardConfig) -> pd.DataFrame:
        if isinstance(data, pd.Series):
            data = data.to_frame(data.name or 'Time series')

        if data.columns.nlevels == 1:
            data.columns.name = data.columns.name or 'variable'
            data = DataProcessor._insert_empty_column_index_level(data)

        if config.facet_col in [data.columns.names[0]]:
            data.columns = data.columns.reorder_levels([1, 0])

        return data

    @staticmethod
    def update_facet_config(data: pd.DataFrame, config: DashboardConfig) -> None:
        unique_facet_col_keys = data.columns.get_level_values(config.facet_col).unique().to_list()
        if config.facet_col_order is None:
            config.facet_col_order = unique_facet_col_keys
        else:
            config.facet_col_order += [c for c in unique_facet_col_keys if c not in config.facet_col_order]

        unique_facet_row_keys = data.columns.get_level_values(config.facet_row).unique().to_list()
        if config.facet_row_order is None:
            config.facet_row_order = unique_facet_row_keys
        else:
            config.facet_row_order += [c for c in unique_facet_row_keys if c not in config.facet_row_order]

        if config.facet_col_wrap is None:
            config.facet_col_wrap = len(config.facet_col_order)

    @staticmethod
    def get_grouped_data(series: pd.Series, x_axis: str, groupby_aggregation: str) -> pd.DataFrame:
        """Group and aggregate time series data"""
        temp = series.to_frame('value')
        temp.loc[:, 'time'] = temp.index.time
        temp.loc[:, 'minute'] = temp.index.minute
        temp.loc[:, 'hour'] = temp.index.hour + 1
        temp.loc[:, 'date'] = temp.index.date
        temp.loc[:, 'month'] = temp.index.month
        temp.loc[:, 'week'] = temp.index.isocalendar().week
        temp.loc[:, 'year_month'] = temp.index.strftime('%Y-%m')
        temp.loc[:, 'year_week'] = temp.index.strftime('%Y-CW%U')

        y_axis = 'time'
        groupby = [y_axis, x_axis]
        temp = temp.groupby(groupby)['value'].agg(groupby_aggregation)
        temp = temp.unstack(x_axis)
        temp_data = temp.sort_index(ascending=False)
        return temp_data

    @staticmethod
    def _insert_empty_column_index_level(df: pd.DataFrame, level_name: str = None) -> pd.DataFrame:
        level_value = ''
        return pd.concat({level_value: df}, axis=1, names=[level_name])

    @staticmethod
    def _prepend_empty_row(df: pd.DataFrame) -> pd.DataFrame:
        empty_row = pd.DataFrame([[np.nan] * len(df.columns)], index=[' '], columns=df.columns)
        return pd.concat([empty_row, df])


class ColorManager:
    @staticmethod
    def get_color_settings_for_facet_category(config: DashboardConfig, facet_key: tuple[str, str]):
        row_key, col_key = facet_key
        settings = {
            'color_continuous_scale': config.figure_kwargs.get('color_continuous_scale'),
            'color_continuous_midpoint': config.figure_kwargs.get('color_continuous_midpoint'),
            'range_color': config.figure_kwargs.get('range_color')
        }

        if config.per_facet_row_colorscale and row_key in config.facet_row_color_settings:
            settings.update(config.facet_row_color_settings.get(row_key, {}))
        elif config.per_facet_col_colorscale and col_key in config.facet_col_color_settings:
            settings.update(config.facet_col_color_settings.get(col_key, {}))

        return settings

    @staticmethod
    def compute_color_params(data, config: DashboardConfig, facet_key: tuple[str, str] = None):
        if facet_key is not None:
            settings = ColorManager.get_color_settings_for_facet_category(config, facet_key)
        else:
            settings = config.figure_kwargs

        if facet_key is not None:
            if config.per_facet_row_colorscale:
                row_key, _ = facet_key
                filtered_data = data.loc[:, (row_key, slice(None))]
            elif config.per_facet_col_colorscale:
                _, col_key = facet_key
                filtered_data = data.loc[:, (slice(None), col_key)]
            else:
                filtered_data = data
        else:
            filtered_data = data

        color_continuous_scale = settings.get('color_continuous_scale')
        color_continuous_midpoint = settings.get('color_continuous_midpoint')
        range_color = settings.get('range_color')

        result = {}
        if color_continuous_scale:
            result['colorscale'] = color_continuous_scale

        if range_color:
            result['zmin'] = range_color[0]
            result['zmax'] = range_color[1]
        elif color_continuous_midpoint == 0:
            _absmax = filtered_data.abs().max().max()
            result['zmin'] = -_absmax
            result['zmax'] = _absmax
        elif color_continuous_midpoint:
            raise NotImplementedError("color_continuous_midpoint other than 0 is not implemented")
        else:
            result['zmin'] = filtered_data.min().min()
            result['zmax'] = filtered_data.max().max()

        return result


class TraceGenerator:
    @staticmethod
    def get_heatmap_trace(data: pd.DataFrame, ts_kwargs, color_kwargs, **kwargs):
        if set(data.columns).issubset(list(range(1, 13))):
            x = [calendar.month_abbr[m] for m in range(1, 13)]
        else:
            x = data.columns

        trace_kwargs = {**color_kwargs, **ts_kwargs, **kwargs}

        assert 'colorscale' in trace_kwargs
        assert 'zmin' in trace_kwargs
        assert 'zmax' in trace_kwargs

        trace_heatmap = go.Heatmap(
            x=x,
            z=data.values,
            y=data.index,
            **trace_kwargs
        )
        return trace_heatmap

    @staticmethod
    def get_stats_trace(series: pd.Series, stat_aggs, stat_kwargs, color_kwargs, **kwargs):
        data_stats = pd.Series({agg: func(series) for agg, func in stat_aggs.items()})
        data_stats = data_stats.to_frame('stats')
        data_stats = DataProcessor._prepend_empty_row(data_stats)

        if 'ygap' not in stat_kwargs:
            stat_kwargs['ygap'] = 5

        text_data = data_stats.map(lambda x: f'{x:.0f}')
        text_data = text_data.replace('nan', '').replace('null', '')

        trace_kwargs = {**color_kwargs, **stat_kwargs, **kwargs}
        trace_kwargs['showscale'] = False  # Stats should never have a colorbar

        assert 'colorscale' in trace_kwargs
        assert 'zmin' in trace_kwargs
        assert 'zmax' in trace_kwargs

        trace_stats = go.Heatmap(
            z=data_stats.values,
            x=data_stats.columns,
            y=data_stats.index,
            text=text_data.values,
            texttemplate="%{text}",
            **trace_kwargs
        )
        return trace_stats

    @staticmethod
    def create_colorscale_trace(z_min, z_max, colorscale, orientation='v', title=None):
        if orientation == 'v':
            z_vals = np.linspace(z_min, z_max, 100).reshape(-1, 1)
        else:
            z_vals = np.linspace(z_min, z_max, 100).reshape(1, -1)

        axis_vals = np.linspace(z_min, z_max, 100)

        colorbar_settings = {
            'thickness': 15,
            'title': title or ''
        }

        if orientation == 'h':
            colorbar_settings.update({
                'orientation': 'h',
                'y': -0.15,
                'xanchor': 'center',
                'x': 0.5
            })
            x = axis_vals
            y = None
        else:
            x = None
            y = axis_vals

        return go.Heatmap(
            x=x,
            y=y,
            z=z_vals,
            colorscale=colorscale,
            showscale=False,
            zmin=z_min,
            zmax=z_max,
            colorbar=colorbar_settings
        )


class TimeSeriesDashboardGenerator:
    def __init__(
            self,
            x_axis: X_AXIS_TYPES = 'date',
            facet_col: str = None,
            facet_row: str = None,
            facet_col_wrap: int = None,
            facet_col_order: list[str] = None,
            facet_row_order: list[str] = None,
            ratio_of_stat_col: float = 0.1,
            stat_aggs: Dict[str, Callable[[pd.Series], float | int]] = None,
            groupby_aggregation: GROUPBY_AGG_TYPES = 'mean',
            title: str = None,
            color_continuous_scale: str | list[str] | list[tuple[float, str]] = None,
            color_continuous_midpoint: int | float = None,
            range_color: list[int | float] = None,
            per_facet_col_colorscale: bool = False,
            per_facet_row_colorscale: bool = False,
            facet_row_color_settings: dict = None,
            facet_col_color_settings: dict = None,
            subplots_vertical_spacing: float = None,
            subplots_horizontal_spacing: float = None,
            time_series_figure_kwargs: dict = None,
            stat_figure_kwargs: dict = None,
            universal_figure_kwargs: dict = None,
            **figure_kwargs
    ):
        self.config = DashboardConfig(
            x_axis=x_axis,
            facet_col=facet_col,
            facet_row=facet_row,
            facet_col_wrap=facet_col_wrap,
            facet_col_order=facet_col_order,
            facet_row_order=facet_row_order,
            ratio_of_stat_col=ratio_of_stat_col,
            stat_aggs=stat_aggs,
            groupby_aggregation=groupby_aggregation,
            title=title,
            color_continuous_scale=color_continuous_scale,
            color_continuous_midpoint=color_continuous_midpoint,
            range_color=range_color,
            per_facet_col_colorscale=per_facet_col_colorscale,
            per_facet_row_colorscale=per_facet_row_colorscale,
            facet_row_color_settings=facet_row_color_settings,
            facet_col_color_settings=facet_col_color_settings,
            subplots_vertical_spacing=subplots_vertical_spacing,
            subplots_horizontal_spacing=subplots_horizontal_spacing,
            time_series_figure_kwargs=time_series_figure_kwargs,
            stat_figure_kwargs=stat_figure_kwargs,
            universal_figure_kwargs=universal_figure_kwargs,
            ** figure_kwargs
        )

    def get_figure(self, data: pd.DataFrame, **kwargs):
        original_config = copy.deepcopy(self.config)

        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

        if not kwargs.get('_skip_validation', False):
            DataProcessor.validate_input_data_and_config(data, self.config)
        data = DataProcessor.prepare_dataframe_for_facet(data, self.config)
        data = DataProcessor.ensure_df_has_two_column_levels(data, self.config)
        DataProcessor.update_facet_config(data, self.config)

        fig = self._create_figure_layout_with_subplots(data)

        self._add_heatmap_and_stat_traces_to_figure(data, fig)

        if self.config.per_facet_col_colorscale:
            self._add_column_colorscales(data, fig)
            fig.update_traces(showlegend=False)
        elif self.config.per_facet_row_colorscale:
            self._add_row_colorscales(data, fig)
            fig.update_traces(showlegend=False)

        if self.config.title:
            fig.update_layout(
                title=f'<b>{self.config.title}</b>',
                title_x=0.5,
            )

        self.config = original_config

        return fig

    def get_figures_chunked(
            self,
            data: pd.DataFrame,
            max_n_rows_per_figure: int = None,
            n_figures: int = None,
            chunk_title_suffix: bool = True,
            **kwargs
    ) -> list[go.Figure]:
        """
        Generate multiple figures by splitting facet rows into chunks.
        """
        original_config = copy.deepcopy(self.config)

        if sum(x is not None for x in [max_n_rows_per_figure, n_figures]) != 1:
            raise ValueError("Provide exactly one of: max_n_rows_per_figure or n_figures")

        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

        data = DataProcessor.prepare_dataframe_for_facet(data, self.config)
        data = DataProcessor.ensure_df_has_two_column_levels(data, self.config)
        DataProcessor.update_facet_config(data, self.config)

        if self.config.facet_row is None:
            return [self.get_figure(data, **kwargs)]

        total_rows = len(self.config.facet_row_order)

        if max_n_rows_per_figure:
            n_chunks = math.ceil(total_rows / max_n_rows_per_figure)
            chunk_size = max_n_rows_per_figure
        else:
            n_chunks = n_figures
            chunk_size = math.ceil(total_rows / n_figures)

        figures = []
        original_title = self.config.title

        for i in range(n_chunks):
            start_idx = i * chunk_size
            end_idx = min(start_idx + chunk_size, total_rows)
            chunk_rows = self.config.facet_row_order[start_idx:end_idx]

            if not chunk_rows:
                continue

            chunk_kwargs = kwargs.copy()
            chunk_kwargs['facet_row_order'] = chunk_rows

            if chunk_title_suffix and original_title:
                chunk_kwargs['title'] = f"{original_title} (Part {i + 1}/{n_chunks})"
            elif chunk_title_suffix:
                chunk_kwargs['title'] = f"Part {i + 1}/{n_chunks}"

            cols_in_chunk = [
                c for c, facet_row_category in
                zip(data.columns, data.columns.get_level_values(self.config.facet_row))
                if facet_row_category in chunk_rows
            ]
            data_chunk = data[cols_in_chunk]

            fig = self.get_figure(data_chunk, **chunk_kwargs, _skip_validation=True)
            figures.append(fig)

        self.config = original_config

        return figures

    def _create_figure_layout_with_subplots(self, data: pd.DataFrame) -> go.Figure:
        facet_col_wrap = max([1, self.config.facet_col_wrap])
        ratio_of_stat_col = self.config.ratio_of_stat_col

        has_colorscale_col = self.config.per_facet_row_colorscale
        has_colorscale_row = self.config.per_facet_col_colorscale

        num_facet_rows = max([1, len(self.config.facet_row_order)])
        num_facet_cols = max([1, len(self.config.facet_col_order)])

        num_rows = math.ceil(num_facet_cols / facet_col_wrap) * num_facet_rows
        num_cols = facet_col_wrap * 2  # Each facet gets a heatmap + stats column

        if has_colorscale_col:
            num_cols += 1
        if has_colorscale_row:
            num_rows += 1

        subplot_titles = self._generate_subplot_titles(has_colorscale_col, has_colorscale_row)
        column_widths = self._get_column_widths(facet_col_wrap, has_colorscale_col, ratio_of_stat_col)
        row_heights = self._get_row_heights(has_colorscale_row, num_rows)
        specs = [[{} for _ in range(num_cols)] for _ in range(num_rows)]

        fig = make_subplots(
            rows=num_rows,
            cols=num_cols,
            subplot_titles=subplot_titles,
            column_widths=column_widths,
            row_heights=row_heights,
            specs=specs,
            vertical_spacing=self.config.subplots_vertical_spacing,
            horizontal_spacing=self.config.subplots_horizontal_spacing,
        )
        fig.update_layout(
            plot_bgcolor='rgba(0, 0, 0, 0)',
            margin=dict(t=50, b=50)
        )

        return fig

    def _get_row_heights(self, has_colorscale_row: bool, num_rows: int) -> list[float]:
        row_heights = None
        if has_colorscale_row:
            regular_height = 1.0
            colorscale_height = 0.15

            total_regular_rows = num_rows - 1
            total_height = total_regular_rows * regular_height + colorscale_height
            norm_regular = regular_height / total_height
            norm_colorscale = colorscale_height / total_height

            row_heights = [norm_regular] * total_regular_rows + [norm_colorscale]
        return row_heights

    def _get_column_widths(self, facet_col_wrap, has_colorscale_col, ratio_of_stat_col) -> list[float]:
        if has_colorscale_col:
            colorscale_width = 0.03
            adjusted_width = 1 - colorscale_width
            column_widths = []

            for _ in range(facet_col_wrap):
                heatmap_width = (adjusted_width - ratio_of_stat_col) / facet_col_wrap
                stats_width = ratio_of_stat_col / facet_col_wrap
                column_widths.extend([heatmap_width, stats_width])

            column_widths.append(colorscale_width)
        else:
            column_widths = [(1 - ratio_of_stat_col) / facet_col_wrap, ratio_of_stat_col / facet_col_wrap] * facet_col_wrap
        return column_widths

    def _generate_subplot_titles(self, has_colorscale_col, has_colorscale_row):
        subplot_titles = []
        for row_name in self.config.facet_row_order:
            for col_name in self.config.facet_col_order:
                if row_name and col_name:
                    title = f'{row_name} - {col_name}'
                else:
                    title = row_name or col_name
                subplot_titles.append(title)  # Title for heatmap
                subplot_titles.append(None)  # Title for stats

            if has_colorscale_col:
                subplot_titles.append(row_name)

        if has_colorscale_row:
            for col_name in self.config.facet_col_order:
                subplot_titles.append(col_name)
                subplot_titles.append(None)
        return subplot_titles

    def _add_heatmap_and_stat_traces_to_figure(self, data, fig):
        facet_col_wrap = self.config.facet_col_wrap

        disable_main_colorbars = self.config.per_facet_col_colorscale or self.config.per_facet_row_colorscale
        if disable_main_colorbars:
            self.config.time_series_figure_kwargs['showscale'] = False

        global_color_params = {}
        if not (self.config.per_facet_col_colorscale or self.config.per_facet_row_colorscale):
            global_color_params = ColorManager.compute_color_params(data, self.config)

        current_row = 1
        row_offset = 0

        for row_idx, row_key in enumerate(self.config.facet_row_order):
            col_offset = 0
            for col_idx, col_key in enumerate(self.config.facet_col_order):
                facet_pos = col_idx % facet_col_wrap
                if facet_pos == 0 and col_idx > 0:
                    row_offset += 1

                fig_row = current_row + row_offset
                fig_col = col_offset + facet_pos * 2 + 1  # +1 because plotly indexing starts at 1

                data_col = facet_key = (row_key, col_key)
                if data_col not in data.columns:
                    continue
                series = data[data_col]

                x_axis = self._get_effective_param_for_data_col('x_axis', data_col)
                groupby_aggregation = self._get_effective_param_for_data_col('groupby_aggregation', data_col)

                self._set_hovertemplates(x_axis)

                grouped_data = DataProcessor.get_grouped_data(series, x_axis, groupby_aggregation)

                color_params = self._get_color_params_for_facet(data, facet_key, global_color_params)

                show_colorbar = False
                if not disable_main_colorbars:
                    show_colorbar = (row_idx == 0 and col_idx == 0)

                heatmap_trace = TraceGenerator.get_heatmap_trace(
                    grouped_data,
                    self.config.time_series_figure_kwargs,
                    color_params,
                    showscale=show_colorbar,
                )

                fig.add_trace(heatmap_trace, row=fig_row, col=fig_col)

                fig.update_yaxes(
                    tickvals=[time(hour=h, minute=0) for h in [0, 6, 12, 18]] + [max(grouped_data.index)],
                    ticktext=['0', '6', '12', '18', '24'],
                    row=fig_row,
                    col=fig_col,
                    autorange='reversed',
                )

                if x_axis == 'year_week':
                    fig.update_xaxes(dtick=8, row=fig_row, col=fig_col)

                stats_trace = TraceGenerator.get_stats_trace(
                    series,
                    self.config.stat_aggs,
                    self.config.stat_figure_kwargs,
                    color_params
                )

                fig.add_trace(stats_trace, row=fig_row, col=fig_col + 1)

                fig.update_xaxes(showgrid=False, row=fig_row, col=fig_col + 1)
                fig.update_yaxes(showgrid=False, autorange='reversed', row=fig_row, col=fig_col + 1)

            if col_offset == 0:
                current_row += math.ceil(len(self.config.facet_col_order) / facet_col_wrap)

    def _get_color_params_for_facet(self, data: pd.DataFrame, facet_key: tuple[str, str], global_color_params: dict) -> dict:
        if self.config.per_facet_col_colorscale or self.config.per_facet_row_colorscale:
            color_params = ColorManager.compute_color_params(data, self.config, facet_key)
        else:
            color_params = global_color_params
        return color_params

    def _add_row_colorscales(self, data, fig):
        colorscale_col = self.config.facet_col_wrap * 2 + 1  # Column after all heatmaps and stats

        for row_idx, row_key in enumerate(self.config.facet_row_order):
            row_pos = row_idx * math.ceil(len(self.config.facet_col_order) / self.config.facet_col_wrap) + 1

            facet_key = (row_key, self.config.facet_col_order[0])
            colorscale, z_max, z_min = self._get_color_settings_for_category(data, facet_key)
            colorscale_trace = TraceGenerator.create_colorscale_trace(
                z_min, z_max, colorscale, 'v', row_key
            )

            fig.add_trace(colorscale_trace, row=row_pos, col=colorscale_col)
            fig.update_xaxes(showticklabels=False, showgrid=False, row=row_pos, col=colorscale_col)
            fig.update_yaxes(showticklabels=True, showgrid=False, row=row_pos, col=colorscale_col, side='right')

    def _get_color_settings_for_category(self, data, facet_key):
        color_params = ColorManager.compute_color_params(data, self.config, facet_key)
        colorscale = color_params.get('colorscale', 'viridis')
        z_min = color_params.get('zmin', 0)
        z_max = color_params.get('zmax', 1)
        return colorscale, z_max, z_min

    def _add_column_colorscales(self, data, fig):
        colorscale_row = math.ceil(len(self.config.facet_col_order) / self.config.facet_col_wrap) * len(
            self.config.facet_row_order) + 1

        for col_idx, col_key in enumerate(self.config.facet_col_order):
            col_pos = (col_idx % self.config.facet_col_wrap) * 2 + 1

            facet_key = (self.config.facet_row_order[0], col_key)

            colorscale, z_max, z_min = self._get_color_settings_for_category(data, facet_key)

            colorscale_trace = TraceGenerator.create_colorscale_trace(
                z_min, z_max, colorscale, 'h', col_key
            )

            fig.add_trace(colorscale_trace, row=colorscale_row, col=col_pos)
            fig.update_xaxes(showticklabels=True, showgrid=False, row=colorscale_row, col=col_pos)
            fig.update_yaxes(showticklabels=False, showgrid=False, row=colorscale_row, col=col_pos)

    def _get_effective_param_for_data_col(self, param_name, data_col):
        param_value = getattr(self.config, param_name)
        if not isinstance(param_value, list):
            return param_value
        else:
            return list(set(param_value).intersection(list(data_col)))[0]

    def _set_hovertemplates(self, x_axis):
        ts_kwargs = self.config.time_series_figure_kwargs
        stat_kwargs = self.config.stat_figure_kwargs

        ts_kwargs['hovertemplate'] = f"{x_axis}: %{{x}}<br>Hour of day: %{{y}}<br>Value: %{{z}}<extra></extra>"
        stat_kwargs['hovertemplate'] = f"aggregation: %{{y}}<br>Value: %{{z}}<extra></extra>"


if __name__ == '__main__':
    url = "https://tubcloud.tu-berlin.de/s/pKttFadrbTKSJKF/download/time-series-lecture-2.csv"
    ts_raw = pd.read_csv(url, index_col=0, parse_dates=True).rename_axis('variable', axis=1)
    ts_res = ts_raw[['onwind', 'offwind', 'solar']].copy() * 100  # to percent
    ts_mixed = ts_raw[['prices', 'load', 'solar']].copy()
    ts_mixed['solar'] *= 100  # to percent
    ts_res_scenarios = pd.concat(
        {
            'base': ts_res,
            'scen1': (ts_res/100) ** 0.5 * 100,
            'scen2': (ts_res/100) ** 0.2 * 100
        },
        axis=1,
        names=['dataset']
    )
    ts_res_scenarios = ts_res_scenarios
    ts_res_scenarios = ts_res_scenarios.drop(('scen1', 'offwind'), axis=1)  # Remove this to have "missing-data"

    generator_raw = TimeSeriesDashboardGenerator(
        x_axis='date',
        color_continuous_scale='viridis',
        facet_row='variable',
        facet_row_order=['solar', 'onwind', 'offwind']
    )
    fig_raw = generator_raw.get_figure(ts_res, title='Variables')
    fig_raw.show(renderer='browser')

    # Basic visualization with custom set of stats
    stats = DashboardConfig.DEFAULT_STATISTICS.copy()
    stats.pop('Datums')
    stats.pop('Abs max')
    stats.pop('Abs mean')
    stats['Median'] = DashboardConfig.STATISTICS_LIBRARY['Median']
    stats['% == 0'] = lambda x: (x == 0).sum() / len(x) * 100  # custom agg should be a callable of a pd.Series
    stats['% > 50'] = lambda x: (x > 50).sum() / len(x) * 100  # custom agg should be a callable of a pd.Series

    generator_custom_stats = TimeSeriesDashboardGenerator(
        x_axis='date',  # X-axis per date (other options are aggregations per: week, month, year_week, year_month)
        color_continuous_scale='viridis',  # Color scale
        facet_row='variable',  # Each variable gets its own row
        facet_row_order=['solar', 'onwind', 'offwind'],
        stat_aggs=stats,
    )

    fig_custom_stats = generator_custom_stats.get_figure(ts_res, title='Renewable Generation Patterns with custom stats')
    fig_custom_stats.show(renderer='browser')

    generator_facet_col_wrap = TimeSeriesDashboardGenerator(
        x_axis='date',
        color_continuous_scale='viridis',
        facet_col='variable',
        facet_col_order=['solar', 'onwind', 'offwind'],
        facet_col_wrap=2
    )
    fig_raw_facet_col_wrap = generator_facet_col_wrap.get_figure(ts_res, title='Variables')
    fig_raw_facet_col_wrap.show(renderer='browser')

    generator_res_scenarios = TimeSeriesDashboardGenerator(
        x_axis='date',
        facet_col='dataset',
        facet_row='variable',
        facet_col_order=['base', 'scen1', 'scen2'],
        facet_row_order=['onwind', 'solar', 'offwind'],
        color_continuous_scale='viridis',
    )
    fig_res_scenarios = generator_res_scenarios.get_figure(ts_res_scenarios, title='Variable per Scenario')
    fig_res_scenarios.show(renderer='browser')

    # Define custom color settings for each row
    color_setting_per_res_var = {
        'onwind': {'color_continuous_scale': 'Blues', 'range_color': [0, 100]},
        'solar': {'color_continuous_scale': 'Reds', 'range_color': [0, 90]},
        'offwind': {'color_continuous_scale': 'Greens', 'range_color': [0, 80]},
    }

    # Use different color scales per row
    generator_per_row = TimeSeriesDashboardGenerator(
        x_axis='date',
        facet_col='dataset',
        facet_row='variable',
        facet_col_order=['base', 'scen1', 'scen2'],
        facet_row_order=['onwind', 'solar', 'offwind'],
        per_facet_row_colorscale=True,
        facet_row_color_settings=color_setting_per_res_var
    )
    fig_per_row = generator_per_row.get_figure(ts_res_scenarios, title='Different Color Scale per Row')
    fig_per_row.show(renderer='browser')

    # Alternative: Use different color scales per column
    generator_per_col = TimeSeriesDashboardGenerator(
        x_axis='date',
        facet_col='variable',
        facet_row='dataset',
        facet_col_order=['onwind', 'solar', 'offwind'],
        facet_row_order=['base', 'scen1', 'scen2'],
        per_facet_col_colorscale=True,
        facet_col_color_settings=color_setting_per_res_var
    )
    fig_per_col = generator_per_col.get_figure(ts_res_scenarios, title='Different Color Scale per Column')
    fig_per_col.show(renderer='browser')

    # Example: Multiple x_axis aggregations with per-row colorscales
    generator_facet_col_different_x_axis = TimeSeriesDashboardGenerator(
        x_axis=['date', 'week', 'year_month'],
        color_continuous_scale='viridis',
        facet_col='x_axis',
        facet_row='variable',
        facet_row_order=['solar', 'load', 'prices'],
        per_facet_row_colorscale=True,
        facet_row_color_settings={
            'load': {'color_continuous_scale': 'Blues'},
            'prices': {'color_continuous_scale': 'Portland', 'color_continuous_midpoint': 0},
            'solar': {'color_continuous_scale': 'Reds', 'range_color': [0, 100]}
        }
    )
    fig_raw_facet_col_x_axis = generator_facet_col_different_x_axis.get_figure(ts_mixed, title='Variables')
    fig_raw_facet_col_x_axis.show(renderer='browser')

    # Example: Multiple groupby-aggs
    generator_aggs = TimeSeriesDashboardGenerator(
        x_axis='month',
        facet_col='groupby_aggregation',
        groupby_aggregation=['min', 'mean', 'max'],
        color_continuous_scale='viridis',
        facet_row='variable',
        facet_row_order=['solar', 'onwind', 'offwind'],
    )
    fig_aggs = generator_aggs.get_figure(ts_res, title='Various Groupby Aggs per column')
    fig_aggs.show(renderer='browser')

    # Example: Multiple groupby-aggs with per-col colorscales
    generator_agg_per_col = TimeSeriesDashboardGenerator(
        x_axis='month',
        facet_col='groupby_aggregation',
        groupby_aggregation=['min', 'mean', 'max'],
        facet_row='variable',
        facet_row_order=['solar', 'onwind', 'offwind'],
        per_facet_col_colorscale=True,
        facet_col_color_settings={
            'min': {'color_continuous_scale': 'Blues', 'range_color': [0, 10]},
            'mean': {'color_continuous_scale': 'Greens', 'range_color': [0, 70]},
            'max': {'color_continuous_scale': 'Reds', 'range_color': [0, 100]}
        },
        subplots_horizontal_spacing=0.05,
    )
    fig_agg_per_col = generator_agg_per_col.get_figure(ts_res, title='Variables')
    fig_agg_per_col.show(renderer='browser')
