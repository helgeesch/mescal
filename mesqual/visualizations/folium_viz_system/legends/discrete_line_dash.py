from mesqual.visualizations.folium_viz_system.legends.base_discrete import DiscreteLegendBase
from mesqual.visualizations.value_mapping_system.discrete import DiscreteLineDashPatternMapping


class DiscreteLineDashLegend(DiscreteLegendBase[DiscreteLineDashPatternMapping]):
    """Legend for line dash pattern mappings"""

    def __init__(
            self,
            mapping: DiscreteLineDashPatternMapping,
            line_color: str = "#000000",
            line_width: int = 2,
            **kwargs
    ):
        super().__init__(mapping, **kwargs)
        self.line_color = line_color
        self.line_width = line_width

    def specific_visual_styles(self) -> str:
        id_selector = f"#{self.get_name()}"
        return f"""
            {id_selector} svg {{
                /* you can put shared SVG styling here if needed */
            }}
        """

    def create_visual_element(self, pattern: str) -> str:
        dash_array = f'stroke-dasharray="{pattern}"' if pattern else ""
        return f"""
            <svg width="{self.visual_column_width}" height="{self.line_width * 2}">
                <line x1="0" y1="{self.line_width}" x2="{self.visual_column_width}" y2="{self.line_width}"
                      stroke="{self.line_color}" stroke-width="{self.line_width}" {dash_array}/>
            </svg>
        """


if __name__ == '__main__':
    import folium

    dash_mapping = DiscreteLineDashPatternMapping(
        mapping={"Type 1": "", "Type 2": "5,5", "Type 3": "10,5,5,5"}
    )
    dash_legend = DiscreteLineDashLegend(
        mapping=dash_mapping,
        title="Line Types",
        visual_column_width=40,
        position={"top": 50, "left": 50},
        width=200
    )

    m = folium.Map(location=[48.85, 2.35], zoom_start=10)
    m.add_child(dash_legend)
    m.save("_tmp/map.html")
