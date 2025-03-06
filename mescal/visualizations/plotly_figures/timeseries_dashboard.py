from typing import Union, List, Literal, Callable, Dict, Any
from itertools import product
from datetime import time
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
    """Configuration class to store and validate dashboard parameters"""

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
            x_axis='date',
            facet_col=None,
            facet_row=None,
            facet_col_wrap=None,
            facet_col_order=None,
            facet_row_order=None,
            ratio_of_stat_col=0.1,
            stat_aggs=None,
            groupby_aggregation='mean',
            title=None,
            color_continuous_scale=None,
            color_continuous_midpoint=None,
            range_color=None,
            per_facet_col_colorscale=False,
            per_facet_row_colorscale=False,
            facet_row_color_settings=None,
            facet_col_color_settings=None,
            time_series_figure_kwargs=None,
            stat_figure_kwargs=None,
            universal_figure_kwargs=None,
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

        # Color settings per facet
        self.facet_row_color_settings = facet_row_color_settings or {}
        self.facet_col_color_settings = facet_col_color_settings or {}

        # Figure kwargs
        self.time_series_figure_kwargs = time_series_figure_kwargs or {}
        self.stat_figure_kwargs = stat_figure_kwargs or {}

        universal_figure_kwargs = universal_figure_kwargs or {}

        self.figure_kwargs = {
            'color_continuous_scale': color_continuous_scale,
            'color_continuous_midpoint': color_continuous_midpoint,
            'range_color': range_color,
            **universal_figure_kwargs,
            **figure_kwargs,
        }


class DataProcessor:
    """Handles data processing and validation for the dashboard"""

    @staticmethod
    def validate_input_data(data: pd.DataFrame, config: DashboardConfig) -> None:
        """Validate input data against configuration parameters"""
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
        """Prepare dataframe for faceting operations"""
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
    def ensure_two_column_levels(data: pd.DataFrame, config: DashboardConfig) -> pd.DataFrame:
        """Ensure dataframe has two column levels"""
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
        """Update facet configuration based on data"""
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
        """Insert an empty column index level"""
        level_value = ''
        return pd.concat({level_value: df}, axis=1, names=[level_name])

    @staticmethod
    def _prepend_empty_row(df: pd.DataFrame) -> pd.DataFrame:
        """Prepend an empty row to a dataframe"""
        empty_row = pd.DataFrame([[np.nan] * len(df.columns)], index=[' '], columns=df.columns)
        return pd.concat([empty_row, df])


class ColorManager:
    """Manages color scales for the dashboard"""

    @staticmethod
    def get_facet_color_settings(config, facet_key):
        """Get color settings for a specific facet"""
        row_key, col_key = facet_key

        # Default settings from global config
        settings = {
            'color_continuous_scale': config.figure_kwargs.get('color_continuous_scale'),
            'color_continuous_midpoint': config.figure_kwargs.get('color_continuous_midpoint'),
            'range_color': config.figure_kwargs.get('range_color')
        }

        # Override with facet-specific settings if available
        if config.per_facet_row_colorscale and row_key in config.facet_row_color_settings:
            settings.update(config.facet_row_color_settings.get(row_key, {}))
        elif config.per_facet_col_colorscale and col_key in config.facet_col_color_settings:
            settings.update(config.facet_col_color_settings.get(col_key, {}))

        return settings

    @staticmethod
    def compute_color_params(data, config, facet_key=None):
        """Compute color parameters for heatmap"""
        # Get settings for this facet
        if facet_key is not None:
            settings = ColorManager.get_facet_color_settings(config, facet_key)
        else:
            settings = config.figure_kwargs

        # If we're using per-facet colorscales, filter data for this facet
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

        # Apply color parameters
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
    """Generates traces for the dashboard"""

    @staticmethod
    def get_heatmap_trace(data: pd.DataFrame, ts_kwargs, color_kwargs, **kwargs):
        """Generate a heatmap trace"""
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
        """Generate a statistics trace"""
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
        """Create a standalone colorscale trace"""
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
            x = axis_vals
            y = None
            colorbar_settings.update({
                'orientation': 'h',
                'y': -0.15,
                'xanchor': 'center',
                'x': 0.5
            })
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
    """Generates time series dashboards with heatmaps and statistics"""

    def __init__(self, **kwargs):
        self.config = DashboardConfig(**kwargs)

    def get_figure(self, data, **kwargs):
        """Generate a dashboard figure for the given data"""
        # Update config with any overrides
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

        # Prepare data and validate
        DataProcessor.validate_input_data(data, self.config)
        data = DataProcessor.prepare_dataframe_for_facet(data, self.config)
        data = DataProcessor.ensure_two_column_levels(data, self.config)
        DataProcessor.update_facet_config(data, self.config)

        # Create figure layout with space for colorscales if needed
        fig = self._create_figure_layout(data)

        # Generate traces for each facet
        self._add_traces_to_figure(data, fig)

        # Add colorscale traces if using per-facet colorscales
        if self.config.per_facet_col_colorscale:
            self._add_column_colorscales(data, fig)
            fig.update_traces(showlegend=False)
        elif self.config.per_facet_row_colorscale:
            self._add_row_colorscales(data, fig)
            fig.update_traces(showlegend=False)

        # Add title if specified
        if self.config.title:
            fig.update_layout(
                title=f'<b>{self.config.title}</b>',
                title_x=0.5,
            )

        return fig

    def _create_figure_layout(self, data):
        """Create the figure layout with subplots"""
        facet_col_wrap = self.config.facet_col_wrap
        ratio_of_stat_col = self.config.ratio_of_stat_col

        # Determine if we need extra space for colorscales
        has_colorscale_col = self.config.per_facet_row_colorscale
        has_colorscale_row = self.config.per_facet_col_colorscale

        # Calculate number of rows and columns needed
        num_facet_rows = len(self.config.facet_row_order)
        num_facet_cols = len(self.config.facet_col_order)

        # Calculate grid dimensions
        num_rows = math.ceil(num_facet_cols / facet_col_wrap) * num_facet_rows
        num_cols = facet_col_wrap * 2  # Each facet gets a heatmap + stats column

        # Add space for colorscales
        if has_colorscale_col:
            num_cols += 1  # Add one column for row colorscales
        if has_colorscale_row:
            num_rows += 1  # Add one row for column colorscales

        # Generate subplot titles (excluding colorscale areas)
        subplot_titles = []
        for row_name in self.config.facet_row_order:
            for col_name in self.config.facet_col_order:
                if row_name and col_name:
                    title = f'{row_name} - {col_name}'
                else:
                    title = row_name or col_name
                subplot_titles.append(title)  # Title for heatmap
                subplot_titles.append(None)  # None for stats

            # Add None for colorscale column if needed
            if has_colorscale_col:
                subplot_titles.append(row_name)

        # Add None titles for colorscale row if needed
        if has_colorscale_row:
            for col_name in self.config.facet_col_order:
                subplot_titles.append(col_name)
                subplot_titles.append(None)

        # Adjust column widths to accommodate colorscales
        if has_colorscale_col:
            colorscale_width = 0.03  # Width of colorscale column
            adjusted_width = 1 - colorscale_width
            column_widths = []

            for _ in range(facet_col_wrap):
                column_widths.extend([
                    (adjusted_width - ratio_of_stat_col) / facet_col_wrap,  # Heatmap
                    ratio_of_stat_col / facet_col_wrap  # Stats
                ])

            # Add width for colorscale column
            column_widths.append(colorscale_width)
        else:
            column_widths = [(1 - ratio_of_stat_col) / facet_col_wrap,
                             ratio_of_stat_col / facet_col_wrap] * facet_col_wrap

        # Create the figure with appropriate specs
        specs = [[{} for _ in range(num_cols)] for _ in range(num_rows)]

        # Set row heights if we have a colorscale row
        row_heights = None
        if has_colorscale_row:
            # Make colorscale row smaller (about 15% of a regular row)
            regular_height = 1.0
            colorscale_height = 0.15

            # Calculate normalized heights
            total_regular_rows = num_rows - 1
            total_height = total_regular_rows * regular_height + colorscale_height
            norm_regular = regular_height / total_height
            norm_colorscale = colorscale_height / total_height

            # Create row heights list with equal heights for data rows and smaller height for colorscale row
            row_heights = [norm_regular] * total_regular_rows + [norm_colorscale]

        # Create figure with adjusted dimensions
        fig = make_subplots(
            rows=num_rows,
            cols=num_cols,
            subplot_titles=subplot_titles,
            column_widths=column_widths,
            row_heights=row_heights,
            specs=specs
        )
        fig.update_layout(
            plot_bgcolor='rgba(0, 0, 0, 0)',
            margin=dict(t=50, b=50)  # Add some margin for better spacing
        )

        return fig

    def _add_traces_to_figure(self, data, fig):
        """Add heatmap and statistics traces to the figure"""
        facet_col_wrap = self.config.facet_col_wrap

        # Make sure to explicitly disable colorbars for all heatmaps
        # when using per-facet colorscales
        disable_main_colorbars = self.config.per_facet_col_colorscale or self.config.per_facet_row_colorscale

        # Configure time series figure settings to disable colorbars if needed
        if disable_main_colorbars:
            self.config.time_series_figure_kwargs['showscale'] = False

        # Compute global color parameters if not using per-facet colors
        global_color_params = {}
        if not (self.config.per_facet_col_colorscale or self.config.per_facet_row_colorscale):
            global_color_params = ColorManager.compute_color_params(data, self.config)

        # Track current subplot position
        current_row = 1
        row_offset = 0

        # Loop through all row facets
        for row_idx, row_key in enumerate(self.config.facet_row_order):
            col_offset = 0

            # Loop through all column facets
            for col_idx, col_key in enumerate(self.config.facet_col_order):
                # Calculate grid position based on facet_col_wrap
                facet_pos = col_idx % facet_col_wrap
                if facet_pos == 0 and col_idx > 0:
                    row_offset += 1

                fig_row = current_row + row_offset
                fig_col = col_offset + facet_pos * 2 + 1  # +1 because plotly indexing starts at 1

                # Get data for this facet
                data_col = (row_key, col_key)
                if data_col not in data.columns:
                    continue

                series = data[data_col]
                facet_key = (row_key, col_key)

                # Determine which x_axis and groupby_aggregation to use
                x_axis = self._get_effective_param('x_axis', data_col)
                groupby_aggregation = self._get_effective_param('groupby_aggregation', data_col)

                # Set hovertemplates
                self._set_hovertemplates(x_axis)

                # Process data for this facet
                grouped_data = DataProcessor.get_grouped_data(series, x_axis, groupby_aggregation)

                # Compute color parameters for this facet
                if self.config.per_facet_col_colorscale or self.config.per_facet_row_colorscale:
                    color_params = ColorManager.compute_color_params(data, self.config, facet_key)
                else:
                    color_params = global_color_params

                # No colorbars on main heatmaps when using separate colorscales
                show_colorbar = False
                if not disable_main_colorbars:
                    show_colorbar = (row_idx == 0 and col_idx == 0)

                # Generate and add heatmap trace
                heatmap_trace = TraceGenerator.get_heatmap_trace(
                    grouped_data,
                    self.config.time_series_figure_kwargs,
                    color_params,
                    showscale=show_colorbar,
                )

                fig.add_trace(heatmap_trace, row=fig_row, col=fig_col)

                # Update y-axis for heatmap
                fig.update_yaxes(
                    tickvals=[time(hour=h, minute=0) for h in [0, 6, 12, 18]] + [max(grouped_data.index)],
                    ticktext=['0', '6', '12', '18', '24'],
                    row=fig_row,
                    col=fig_col,
                    autorange='reversed',
                )

                # Special case for year_week x_axis
                if x_axis == 'year_week':
                    fig.update_xaxes(dtick=8, row=fig_row, col=fig_col)

                # Generate and add statistics trace
                stats_trace = TraceGenerator.get_stats_trace(
                    series,
                    self.config.stat_aggs,
                    self.config.stat_figure_kwargs,
                    color_params
                )

                fig.add_trace(stats_trace, row=fig_row, col=fig_col + 1)

                # Update axes for statistics
                fig.update_xaxes(showgrid=False, row=fig_row, col=fig_col + 1)
                fig.update_yaxes(showgrid=False, autorange='reversed', row=fig_row, col=fig_col + 1)

            # Move to next row for next facet_row
            if col_offset == 0:  # Only increment once per facet_row
                current_row += math.ceil(len(self.config.facet_col_order) / facet_col_wrap)

    def _add_row_colorscales(self, data, fig):
        """Add colorscale for each row facet"""
        colorscale_col = self.config.facet_col_wrap * 2 + 1  # Column after all heatmaps and stats

        # Loop through all row facets
        for row_idx, row_key in enumerate(self.config.facet_row_order):
            # Calculate row position for this facet
            row_pos = row_idx * math.ceil(len(self.config.facet_col_order) / self.config.facet_col_wrap) + 1

            # Use first column's data for this row to get color settings
            facet_key = (row_key, self.config.facet_col_order[0])

            # Get color settings for this row
            color_params = ColorManager.compute_color_params(data, self.config, facet_key)
            colorscale = color_params.get('colorscale', 'viridis')
            z_min = color_params.get('zmin', 0)
            z_max = color_params.get('zmax', 1)

            # Create and add colorscale trace
            colorscale_trace = TraceGenerator.create_colorscale_trace(
                z_min, z_max, colorscale, 'v', row_key
            )

            fig.add_trace(colorscale_trace, row=row_pos, col=colorscale_col)

            fig.update_xaxes(showticklabels=False, showgrid=False, row=row_pos, col=colorscale_col)
            fig.update_yaxes(showticklabels=True, showgrid=False, row=row_pos, col=colorscale_col, side='right')

    def _add_column_colorscales(self, data, fig):
        """Add colorscale for each column facet"""
        # Last row is reserved for colorscales
        colorscale_row = math.ceil(len(self.config.facet_col_order) / self.config.facet_col_wrap) * len(
            self.config.facet_row_order) + 1

        # Loop through all column facets
        for col_idx, col_key in enumerate(self.config.facet_col_order):
            # Calculate column position for this facet
            col_pos = (col_idx % self.config.facet_col_wrap) * 2 + 1

            # Use first row's data for this column to get color settings
            facet_key = (self.config.facet_row_order[0], col_key)

            # Get color settings for this column
            color_params = ColorManager.compute_color_params(data, self.config, facet_key)
            colorscale = color_params.get('colorscale', 'viridis')
            z_min = color_params.get('zmin', 0)
            z_max = color_params.get('zmax', 1)

            # Create and add colorscale trace
            colorscale_trace = TraceGenerator.create_colorscale_trace(
                z_min, z_max, colorscale, 'h', col_key
            )

            fig.add_trace(colorscale_trace, row=colorscale_row, col=col_pos)

            # Hide axes
            fig.update_xaxes(showticklabels=True, showgrid=False, row=colorscale_row, col=col_pos)
            fig.update_yaxes(showticklabels=False, showgrid=False, row=colorscale_row, col=col_pos)

    def _get_effective_param(self, param_name, data_col):
        """Get the effective parameter value for a given data column"""
        param_value = getattr(self.config, param_name)
        if not isinstance(param_value, list):
            return param_value
        else:
            # Find the intersection between the parameter list and the data column
            return list(set(param_value).intersection(list(data_col)))[0]

    def _set_hovertemplates(self, x_axis):
        """Set hover templates for heatmap and statistics traces"""
        ts_kwargs = self.config.time_series_figure_kwargs
        stat_kwargs = self.config.stat_figure_kwargs

        ts_kwargs['hovertemplate'] = f"{x_axis}: %{{x}}<br>Hour of day: %{{y}}<br>Value: %{{z}}<extra></extra>"
        stat_kwargs['hovertemplate'] = f"aggregation: %{{y}}<br>Value: %{{z}}<extra></extra>"


if __name__ == '__main__':
    url = "https://tubcloud.tu-berlin.de/s/pKttFadrbTKSJKF/download/time-series-lecture-2.csv"
    ts_raw = pd.read_csv(url, index_col=0, parse_dates=True).rename_axis('variable', axis=1)
    ts_res = ts_raw[['onwind', 'offwind', 'solar']] * 100  # to percent
    ts_mixed = ts_raw[['prices', 'load', 'solar']]
    ts_mixed['solar'] *= 100  # to percent

    generator_raw = TimeSeriesDashboardGenerator(
        x_axis='date',
        color_continuous_scale='viridis',
        facet_row='variable',
        facet_row_order=['solar', 'onwind', 'offwind']
    )
    fig_raw = generator_raw.get_figure(ts_res, title='Variables')
    fig_raw.show(renderer='browser')

    generator_facet_col_wrap = TimeSeriesDashboardGenerator(
        x_axis='date',
        color_continuous_scale='viridis',
        facet_col='variable',
        facet_col_order=['solar', 'onwind', 'offwind'],
        facet_col_wrap=2
    )
    fig_raw_facet_col_wrap = generator_facet_col_wrap.get_figure(ts_res, title='Variables')
    fig_raw_facet_col_wrap.show(renderer='browser')

    # Example: Multiple scenarios with per-row colorscales
    ts_res_scenarios = pd.concat(
        {
            'base': ts_res,
            'scen1': (ts_res/100) ** 0.7 * 100,
            'scen2': (ts_res/100) ** 0.5 * 100
        },
        axis=1,
        names=['dataset']
    )
    ts_res_scenarios = ts_res_scenarios
    ts_res_scenarios = ts_res_scenarios.drop(('scen1', 'offwind'), axis=1)  # Remove this to have a "missing-data" point

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
