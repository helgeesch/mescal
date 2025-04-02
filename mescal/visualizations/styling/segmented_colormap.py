from typing import Dict, List, Tuple, Optional
import numpy as np
import branca.colormap as cm
from branca.element import MacroElement, Template
import folium

from mescal.utils.color_utils.conversion import to_hex


class SegmentedColorMap:
    """Base class handling color scale logic without visualization."""

    def __init__(
            self,
            segments: Dict[Tuple[float, float], List[str]],
            na_color: str = '#FFFFFF'
    ):
        self.segments_dict = segments
        self.sorted_segments = self._sort_segments(segments)
        self.colors, self.index = self._process_colors_and_index()
        self.tick_values = self._get_tick_values()
        self.na_color = na_color
        self.cmap = cm.LinearColormap(
            self.colors,
            index=self.index,
            vmin=self.min_value,
            vmax=self.max_value,
        )

    def _sort_segments(self, segments):
        sorted_segments = sorted(segments.items(), key=lambda x: x[0][0])
        prev_end = -float('inf')
        for (start, end), _ in sorted_segments:
            if start < prev_end:
                raise ValueError(
                    f"Overlapping segments detected: ({start}, {end}) overlaps with previous segment ending at {prev_end}")
            prev_end = end
        return sorted_segments

    def _process_colors_and_index(self):
        colors = []
        index = []
        self.min_value = self.sorted_segments[0][0][0]
        self.max_value = self.sorted_segments[-1][0][1]

        for (start, end), color_segment in self.sorted_segments:
            color_list = [to_hex(c) for c in color_segment]
            num_colors = len(color_list)

            if num_colors == 1:
                index.append(start)
                colors.append(color_list[0])
                index.append(end if end == self.max_value else end - 1e-6)
                colors.append(color_list[0])
            else:
                positions = np.linspace(start, end, num_colors)
                for pos, color in zip(positions, color_list):
                    if pos == end and end != self.max_value:
                        pos -= 1e-6
                    index.append(pos)
                    colors.append(color)
        return colors, index

    def _get_tick_values(self):
        return [seg[0][0] for seg in self.sorted_segments] + [self.sorted_segments[-1][0][1]]

    def __call__(self, value: float) -> str:
        if np.isnan(value) and self.na_color is not None:
            return to_hex(self.na_color)
        return self.cmap(value)

    def to_normalized_colorscale(
            self,
            num_reference_points_per_segment: int = 10
    ) -> List[Tuple[float, str]]:
        """
        Generate a Plotly-compatible colorscale with:
        - Explicit segment boundaries
        - No overlapping positions
        - Epsilon-adjusted edges between segments

        Args:
            num_reference_points_per_segment: Number of points per segment
                                              (including boundaries)

        Returns: List of (position, color) tuples sorted by position
        """
        if num_reference_points_per_segment < 2:
            raise ValueError(f'num_reference_points_per_segment must be >= 2.')
        colorscale = []
        total_range = self.max_value - self.min_value

        for i, ((seg_start, seg_end), _) in enumerate(self.sorted_segments):
            norm_start, norm_end = self._calc_normalized_positions(seg_end, seg_start, total_range)

            if i < len(self.sorted_segments) - 1:
                norm_end = self._adjust_end_for_all_but_last_segment(norm_end)

            positions = np.linspace(norm_start, norm_end, num_reference_points_per_segment)

            for pos in positions:
                value = self.min_value + pos * total_range
                colorscale.append((round(pos, 10), self(value)))

        final_pos = (self.max_value - self.min_value) / total_range
        colorscale.append((final_pos, self(self.max_value)))

        return self._deduplicate_while_preserving_order(colorscale)

    def _deduplicate_while_preserving_order(self, colorscale):
        seen = set()
        return [
            (pos, color[:-2]) for pos, color in sorted(colorscale, key=lambda x: x[0])
            if not (pos in seen or seen.add(pos))
        ]

    def _adjust_end_for_all_but_last_segment(self, norm_end):
        epsilon = 1e-9
        return norm_end - epsilon

    def _calc_normalized_positions(self, seg_end, seg_start, total_range):
        norm_start = (seg_start - self.min_value) / total_range
        norm_end = (seg_end - self.min_value) / total_range
        return norm_start, norm_end


class SegmentedColorMapLegend(SegmentedColorMap, MacroElement):
    _template = Template("""
    {% macro header(this, kwargs) %}
        <style>
            #{{ this.get_name() }} {
                position: fixed;
                z-index: 1000;
                background: {{ this.background_color }};
                padding: {{ this.padding }}px;
                border-radius: 5px;
                box-shadow: 0 0 5px rgba(0,0,0,0.2);
                font-family: Arial, sans-serif;
                {% for key, value in this.position.items() %}
                {{ key }}: {{ value }};
                {% endfor %}
            }
            #{{ this.get_name() }} .colormap-title {
                margin-bottom: 8px;
                font-size: {{ this.title_font_size }}px;
                font-weight: bold;
                color: {{ this.title_color }};
            }
        </style>
    {% endmacro %}

    {% macro html(this, kwargs) %}
        <div id="{{ this.get_name() }}">
            {% if this.title %}
            <div class="colormap-title">{{ this.title }}</div>
            {% endif %}
            <div style="position: relative; height: {{ this.total_height }}px; width: {{ this.width }}px;">
                <div style="position: absolute; top: 0; left: 0; width: 100%; height: {{ this.bar_height }}px;">
                    {% for segment in this.segments_html %}{{ segment }}{% endfor %}
                </div>
                <div style="position: absolute; top: {{ this.bar_height + 2 }}px; left: 0; width: 100%; height: 20px;">
                    {% for tick in this.tick_html %}{{ tick }}{% endfor %}
                </div>
            </div>
        </div>
    {% endmacro %}
    """)

    def __init__(
            self,
            segments: Dict[Tuple[float, float], List[str]],
            title: str,
            na_color: str = '#FFFFFF',
            background_color: str = "white",
            title_color: str = "#333333",
            title_font_size: int = 14,
            tick_color: Optional[str] = None,
            tick_font_size: int = 12,
            width: int = 400,
            bar_height: int = 30,
            total_height: int = 60,
            padding: int = 10,
            position: Optional[Dict[str, int | float | str]] = None,
            n_ticks_per_segment: int = 2
    ):
        SegmentedColorMap.__init__(self, segments, na_color)
        MacroElement.__init__(self)
        self._name = "SegmentedColorMap"

        self.title = title
        self.background_color = background_color
        self.title_color = title_color
        self.title_font_size = title_font_size
        self.tick_color = tick_color or title_color
        self.tick_font_size = tick_font_size
        self.width = width
        self.bar_height = bar_height
        self.total_height = total_height
        self.padding = padding
        self.n_ticks_per_segment = n_ticks_per_segment

        self.position = self._process_position(position or {"bottom": "20px", "left": "20px"})

        self.segments_html, self.tick_html = self._create_html_components()

    def _process_position(self, position: Dict[str, int | float | str]) -> Dict[str, str]:
        return {k: f"{v}px" if isinstance(v, (int, float)) else str(v)
                for k, v in position.items()}

    def _create_html_components(self):
        num_segments = len(self.sorted_segments)
        segment_width_pct = 100.0 / num_segments

        segments_html = self._generate_segments_html(segment_width_pct)
        all_ticks = self._generate_ticks_with_n_ticks_per_segment()
        unique_ticks = self._deduplicate_ticks_while_preserving_order(all_ticks)
        tick_positions = self._calculate_tick_positions(segment_width_pct, unique_ticks)

        tick_html = [
            f'<div style="position: absolute; left: {pos}%; '
            f'transform: translateX(-50%); top: 0; '
            f'font-size: {self.tick_font_size}px; '
            f'color: {self.tick_color};">{self._format_tick(tick)}</div>'
            for pos, tick in zip(tick_positions, unique_ticks)
        ]

        return segments_html, tick_html

    def _calculate_tick_positions(self, segment_width_pct, unique_ticks):
        tick_positions = []
        for tick in unique_ticks:
            for i, ((seg_start, seg_end), _) in enumerate(self.sorted_segments):
                if seg_start <= tick <= seg_end:
                    if seg_end == seg_start:
                        pos_pct = i * segment_width_pct
                    else:
                        progress = (tick - seg_start) / (seg_end - seg_start)
                        pos_pct = (i * segment_width_pct) + (progress * segment_width_pct)
                    tick_positions.append(pos_pct)
                    break
        return tick_positions

    def _deduplicate_ticks_while_preserving_order(self, all_ticks):
        seen = set()
        unique_ticks = []
        for tick in all_ticks:
            key = round(tick, 9)  # Handle floating precision
            if key not in seen:
                seen.add(key)
                unique_ticks.append(tick)
        return unique_ticks

    def _generate_ticks_with_n_ticks_per_segment(self):
        all_ticks = []
        for (seg_start, seg_end), _ in self.sorted_segments:
            ticks = np.linspace(seg_start, seg_end, self.n_ticks_per_segment)
            all_ticks.extend(ticks)
        return all_ticks

    def _generate_segments_html(self, segment_width_pct):
        segments_html = []
        for (start, end), colors in self.sorted_segments:
            color_list = [to_hex(c) for c in colors]
            if len(color_list) == 1:
                gradient = f"linear-gradient(to right, {color_list[0]}, {color_list[0]})"
            else:
                stops = [
                    f"{color} {100 * i / (len(color_list) - 1):.2f}%"
                    for i, color in enumerate(color_list)
                ]
                gradient = f"linear-gradient(to right, {', '.join(stops)})"

            segments_html.append(
                f'<div style="width: {segment_width_pct}%; height: {self.bar_height}px; '
                f'float: left; background: {gradient};"></div>'
            )
        return segments_html

    def _format_tick(self, value: float) -> str:
        """Format tick labels based on value magnitude."""
        if value % 1 == 0:
            return f"{int(value)}"
        return f"{value:.2f}".rstrip('0').rstrip('.')


if __name__ == '__main__':
    segments = {
        (-20, 0): ['#0000FF', '#00FFFF'],
        (0, 5): ['#00FF00'],
        (5, 10): ['#FF0000', '#FFFF00'],
        (10, 20): ['#FFFFFF']
    }
    segments_2 = {
        (-1000, 0): ['#0000FF', '#00FFFF'],
        (0, 5): ['#00FF00'],
        (5, 10): ['#FF0000', '#FFFF00'],
        (10, 20): ['#000000']
    }

    scm = SegmentedColorMapLegend(
        segments=segments,
        title="Elevation Scale (meters)",
        background_color="#f0f0f0",
        title_color="#2a2a2a",
        title_font_size=12,
        width=500,
        bar_height=25,
        total_height=35,
        padding=20,
        position=dict(bottom=50, left=30),
        n_ticks_per_segment=5,
    )

    scm_2 = SegmentedColorMapLegend(
        segments_2,
        title='BingBong',
        position=dict(bottom=50, left=650),
        n_ticks_per_segment=3,
    )

    m = folium.Map(location=[45.5236, -122.6750], zoom_start=13)
    scm.add_to(m)
    scm_2.add_to(m)
    m.save("_tmp/test_colormap.html")

    plotly_cscale = scm.to_normalized_colorscale(num_reference_points_per_segment=2)
    print(plotly_cscale)
