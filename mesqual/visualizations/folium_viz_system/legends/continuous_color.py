from typing import Any

from mesqual.visualizations.folium_viz_system.legends.base_continuous import ContinuousLegendBase
from mesqual.visualizations.value_mapping_system.continuous import SegmentedContinuousColorscale


class ContinuousColorscaleLegend(ContinuousLegendBase[SegmentedContinuousColorscale]):
    """Legend for continuous color mappings"""

    def specific_visual_styles(self) -> str:
        # no extra CSS beyond base
        return ""

    def create_segment_visual(self, start: float, end: float, colors: Any) -> str:
        if isinstance(colors, list):
            if len(colors) == 1:
                gradient = f"background: {colors[0]};"
            else:
                stops = [
                    f"{color} {100 * i / (len(colors) - 1):.1f}%"
                    for i, color in enumerate(colors)
                ]
                gradient = f"background: linear-gradient(to right, {', '.join(stops)});"
        else:
            gradient = f"background: {colors};"
        return f'<div class="segment-visual" style="{gradient}"></div>'


if __name__ == '__main__':
    import folium

    color_segments = {
        (-20, 0): ['#0000FF', '#00FFFF'],
        (0, 5): ['#00FF00'],
        (5, 10): ['#FF0000', '#FFFF00'],
        (10, 20): ['#FFFFFF']
    }
    color_mapping = SegmentedContinuousColorscale(segments=color_segments)
    color_legend = ContinuousColorscaleLegend(
        mapping=color_mapping,
        title="Temperature (°C)",
        background_color="#f0f0f0",
        n_ticks_per_segment=3,
        segment_spacing=5,
        position={"bottom": 50, "left": 50},
        width=400,
        merge_adjacent_ticks=False,
    )

    m = folium.Map(location=[48.85, 2.35], zoom_start=10)
    m.add_child(color_legend)
    m.save("_tmp/map.html")
