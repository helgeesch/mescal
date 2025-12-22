from mesqual.visualizations.folium_viz_system.legends.base_continuous import ContinuousLegendBase
from mesqual.visualizations.value_mapping_system.continuous import SegmentedContinuousLineWidthMapping


class ContinuousLineWidthMapLegend(ContinuousLegendBase[SegmentedContinuousLineWidthMapping]):
    def __init__(
        self,
        mapping: SegmentedContinuousLineWidthMapping,
        line_color: str = "#000000",
        show_pixel_values: bool = False,
        merge_adjacent_ticks: bool = True,
        **kwargs
    ):
        super().__init__(mapping, **kwargs)
        self.line_color = line_color
        self.show_pixel_values = show_pixel_values
        self.merge_adjacent_ticks = merge_adjacent_ticks

    def specific_visual_styles(self) -> str:
        id_selector = f"#{self.get_name()}"
        return f"""
            {id_selector} .pixel-value {{
                font-size: {self.tick_font_size - 2}px;
                color: {self.tick_color};
                margin-bottom: 4px;
            }}
        """

    def create_segment_visual(
            self,
            start: float,
            end: float,
            width_data: float | list[float]
    ) -> str:
        # trapezoid SVG
        if isinstance(width_data, list):
            w0, w1 = width_data[0], width_data[-1]
        else:
            w0 = w1 = width_data

        svg = f"""
            <svg width="100%" height="{self.segment_height}" preserveAspectRatio="none">
                <path d="
                    M0,{self.segment_height / 2 - w0 / 2}
                    L100,{self.segment_height / 2 - w1 / 2}
                    L100,{self.segment_height / 2 + w1 / 2}
                    L0,{self.segment_height / 2 + w0 / 2}
                    Z
                " fill="{self.line_color}" />
            </svg>
        """
        # pixel-value above ticks
        if self.show_pixel_values:
            label = f"{int(w0)}px" if w0 == w1 else f"{int(w0)}-{int(w1)}px"
            svg += f'<div class="pixel-value">{label}</div>'

        return svg


if __name__ == '__main__':
    import folium

    width_segments = {
        (0, 10): 1.0,
        (10, 20): 3.0,
        (20, 30): [3.0, 10.0],
        (30, 50): 10.0
    }
    width_mapping = SegmentedContinuousLineWidthMapping(segments=width_segments)
    width_legend = ContinuousLineWidthMapLegend(
        mapping=width_mapping,
        title="Flow Volume (MW)",
        line_color="#2c3e50",
        segment_spacing=15,
        position={"bottom": 50, "right": 50},
        merge_adjacent_ticks=False,
        show_pixel_values=True,
    )

    m = folium.Map(location=[48.85, 2.35], zoom_start=10)
    m.add_child(width_legend)
    m.save("_tmp/map.html")
