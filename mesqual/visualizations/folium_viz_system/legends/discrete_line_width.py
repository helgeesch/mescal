from mesqual.visualizations.folium_viz_system.legends.base_discrete import DiscreteLegendBase
from mesqual.visualizations.value_mapping_system.discrete import DiscreteLineWidthMapping


class DiscreteLineWidthLegend(DiscreteLegendBase[DiscreteLineWidthMapping]):
    """Legend for discrete line width mappings"""

    def __init__(
            self,
            mapping: DiscreteLineWidthMapping,
            line_color: str = "#000000",
            show_pixel_values: bool = False,
            **kwargs
    ):
        super().__init__(mapping, **kwargs)
        self.line_color = line_color
        self.show_pixel_values = show_pixel_values

    def specific_visual_styles(self) -> str:
        id_selector = f"#{self.get_name()}"
        return f"""
            {id_selector} .pixel-value {{
                font-size: {self.label_font_size - 2}px;
                color: {self.title_color};
                opacity: 0.7;
                margin-left: 5px;
            }}
        """

    def create_visual_element(self, width: float) -> str:
        height = max(10, width * 2)
        return f"""
            <svg width="{self.visual_column_width}" height="{height}">
                <line x1="0" y1="{height / 2}" x2="{self.visual_column_width}" y2="{height / 2}"
                      stroke="{self.line_color}" stroke-width="{width}"/>
            </svg>
        """

    def render_content(self) -> str:
        items = []

        # Override to add pixel values
        for key, width in sorted(self.mapping.mapping.items()):
            visual = self.create_visual_element(width)
            label = self._format_value(key)
            if self.show_pixel_values:
                label += f'<span class="pixel-value">({width}px)</span>'

            items.append(f"""
                <div class="discrete-item">
                    <div class="visual-column">{visual}</div>
                    <div class="label-column">{label}</div>
                </div>
            """)

        if self.mapping.default_output is not None:
            width = self.mapping.default_output
            visual = self.create_visual_element(width)
            label = "Other"
            if self.show_pixel_values:
                label += f'<span class="pixel-value">({width}px)</span>'

            items.append(f"""
                <div class="discrete-item">
                    <div class="visual-column">{visual}</div>
                    <div class="label-column">{label}</div>
                </div>
            """)

        return ''.join(items)


if __name__ == '__main__':
    import folium

    width_mapping = DiscreteLineWidthMapping(
        mapping={"Type 1": 1, "Type 2": 5, "Type 3": 15}
    )
    width_legend = DiscreteLineWidthLegend(
        mapping=width_mapping,
        title="Line Width",
        visual_column_width=40,
        position={"top": 300, "left": 50},
        width=150
    )

    m = folium.Map(location=[48.85, 2.35], zoom_start=10)
    m.add_child(width_legend)
    m.save("_tmp/map.html")
