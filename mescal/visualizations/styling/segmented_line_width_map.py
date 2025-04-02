from typing import Dict, List, Tuple, Optional, Union

import numpy as np
from branca.element import MacroElement, Template


class SegmentedLineWidthMap:
    """Maps values to line widths using customizable segments similar to SegmentedColorMap."""

    def __init__(
            self,
            segments: dict[tuple[float, float], float | list[float]],
            na_width: float = 1.0
    ):
        self.segments_dict = segments
        self.sorted_segments = self._sort_segments(segments)
        self.min_value = self.sorted_segments[0][0][0]
        self.max_value = self.sorted_segments[-1][0][1]
        self.na_width = na_width

    def _sort_segments(self, segments):
        sorted_segments = sorted(segments.items(), key=lambda x: x[0][0])
        prev_end = -float('inf')
        for (start, end), _ in sorted_segments:
            if start < prev_end:
                raise ValueError(
                    f"Overlapping segments detected: ({start}, {end}) overlaps with previous segment ending at {prev_end}")
            prev_end = end
        return sorted_segments

    def __call__(self, value: float) -> float:
        """Get the appropriate width for a value."""
        if np.isnan(value):
            return self.na_width

        for (start, end), width_data in self.sorted_segments:
            if start <= value <= end:
                if isinstance(width_data, (int, float)):
                    return width_data

                width_list = width_data if isinstance(width_data, list) else [width_data]
                if len(width_list) == 1:
                    return width_list[0]

                idx_low, idx_high, pos_idx = self._interpolate_between_width_points(start, end, value, width_list)

                if idx_low == idx_high:
                    return width_list[idx_low]

                return self._linear_interpolation(idx_high, idx_low, pos_idx, width_list)

        if value < self.min_value:
            first_width = self.sorted_segments[0][1]
            return first_width[0] if isinstance(first_width, list) else first_width
        last_width = self.sorted_segments[-1][1]
        return last_width[-1] if isinstance(last_width, list) else last_width

    @staticmethod
    def _linear_interpolation(idx_high, idx_low, pos_idx, width_list):
        width_low = width_list[idx_low]
        width_high = width_list[idx_high]
        frac = pos_idx - idx_low
        return width_low + frac * (width_high - width_low)

    @staticmethod
    def _interpolate_between_width_points(start, end, value, width_list):
        pos_in_segment = (value - start) / (end - start)
        pos_idx = pos_in_segment * (len(width_list) - 1)
        idx_low = int(np.floor(pos_idx))
        idx_high = int(np.ceil(pos_idx))
        return idx_low, idx_high, pos_idx


class SegmentedLineWidthMapLegend(SegmentedLineWidthMap, MacroElement):
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
            #{{ this.get_name() }} .linewidth-title {
                margin-bottom: 8px;
                font-size: {{ this.title_font_size }}px;
                font-weight: bold;
                color: {{ this.title_color }};
            }
            #{{ this.get_name() }} .line-segment {
                height: 0;
                border-style: solid;
                border-color: {{ this.line_color }};
                margin: 10px 0;
            }
        </style>
    {% endmacro %}

    {% macro html(this, kwargs) %}
        <div id="{{ this.get_name() }}">
            {% if this.title %}
            <div class="linewidth-title">{{ this.title }}</div>
            {% endif %}
            <div style="position: relative; width: {{ this.width }}px;">
                {% for segment in this.segments_html %}{{ segment }}{% endfor %}
            </div>
        </div>
    {% endmacro %}
    """)

    def __init__(
            self,
            segments: Dict[Tuple[float, float], Union[float, List[float]]],
            title: str,
            na_width: float = 1.0,
            background_color: str = "white",
            title_color: str = "#333333",
            title_font_size: int = 14,
            line_color: str = "#000000",
            width: int = 400,
            padding: int = 10,
            position: Optional[Dict[str, Union[int, float, str]]] = None,
            tick_color: str = "#333333",
            tick_font_size: int = 12,
            show_pixel_values: bool = False
    ):
        SegmentedLineWidthMap.__init__(self, segments, na_width)
        MacroElement.__init__(self)
        self._name = "LineWidthMapLegend"

        self.title = title
        self.background_color = background_color
        self.title_color = title_color
        self.title_font_size = title_font_size
        self.line_color = line_color
        self.width = width
        self.padding = padding
        self.tick_color = tick_color
        self.tick_font_size = tick_font_size
        self.show_pixel_values = show_pixel_values

        self.position = self._process_position(position or {"bottom": "20px", "right": "20px"})
        self.segments_html = self._create_segments_html()

    def _process_position(self, position: Dict[str, Union[int, float, str]]) -> Dict[str, str]:
        return {k: f"{v}px" if isinstance(v, (int, float)) else str(v)
                for k, v in position.items()}

    def _create_segments_html(self) -> List[str]:
        segments_html = []

        for i, ((start, end), width_data) in enumerate(self.sorted_segments):
            if isinstance(width_data, (int, float)):
                # Same width for start and end
                start_width = end_width = width_data
            else:
                width_list = width_data if isinstance(width_data, list) else [width_data]
                if len(width_list) == 1:
                    # Same width for start and end
                    start_width = end_width = width_list[0]
                else:
                    # Different widths for start and end
                    start_width = width_list[0]
                    end_width = width_list[-1]

            segments_html.append(self._create_segment_html(start, end, start_width, end_width))

        return segments_html

    def _create_segment_html(self, start: float, end: float, start_width: float, end_width: float) -> str:
        """Create HTML for a single line segment with start and end widths."""
        label = self._format_range(start, end)

        pixel_value_html = ""
        if self.show_pixel_values:
            if start_width == end_width:
                pixel_value_html = f"""
                <div style="width: 40px; text-align: right; margin-left: 10px; 
                     font-size: {self.tick_font_size}px; color: {self.tick_color};">{start_width}px</div>
                """
            else:
                pixel_value_html = f"""
                <div style="width: 40px; text-align: right; margin-left: 10px; 
                     font-size: {self.tick_font_size}px; color: {self.tick_color};">{start_width}-{end_width}px</div>
                """

        return f"""
        <div style="display: flex; align-items: center; margin-bottom: 3px;">
            <div style="flex: 1; margin-right: 10px; 
                 font-size: {self.tick_font_size}px; color: {self.tick_color};">{label}</div>
            <div style="flex: 2; display: flex;">
                <div class="line-segment" style="flex: 1; border-width: {start_width}px;"></div>
                <div class="line-segment" style="flex: 1; border-width: {end_width}px;"></div>
            </div>
            {pixel_value_html}
        </div>
        """

    def _format_range(self, start: float, end: float) -> str:
        """Format a value range for display."""
        start_str = self._format_value(start)
        end_str = self._format_value(end)
        return f"{start_str} - {end_str}"

    def _format_value(self, value: float) -> str:
        """Format a single value for display."""
        if value % 1 == 0:
            return f"{int(value)}"
        return f"{value:.2f}".rstrip('0').rstrip('.')


if __name__ == '__main__':
    import folium

    # Example segments for line widths
    width_segments = {
        (0, 10): 1.0,
        (10, 20): 3.0,
        (20, 30): [3.0, 5.0],  # Interpolated from 3 to 5
        (30, 50): 8.0
    }

    width_map = SegmentedLineWidthMapLegend(
        segments=width_segments,
        title="Flow Volume (MW)",
        background_color="#f5f5f5",
        line_color="#112233",
        width=150,
        padding=15,
        position={"bottom": 20, "right": 20},
        show_pixel_values=False  # Set to False by default
    )

    m = folium.Map(location=[48.85, 2.35], zoom_start=10)
    width_map.add_to(m)
    m.save("_tmp/test_line_width_map.html")