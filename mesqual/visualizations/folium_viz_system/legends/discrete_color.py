from mesqual.visualizations.folium_viz_system.legends.base_discrete import DiscreteLegendBase
from mesqual.visualizations.value_mapping_system.discrete import DiscreteColorMapping


class DiscreteColorLegend(DiscreteLegendBase[DiscreteColorMapping]):
    """Legend for discrete color mappings"""

    def __init__(
            self,
            mapping: DiscreteColorMapping,
            swatch_size: int = 20,
            **kwargs
    ):
        super().__init__(mapping, **kwargs)
        self.swatch_size = swatch_size

    def specific_visual_styles(self) -> str:
        id_selector = f"#{self.get_name()}"
        return f"""
            {id_selector} .color-swatch {{
                width: {self.swatch_size}px;
                height: {self.swatch_size}px;
                border: 1px solid #ccc;
                border-radius: 2px;
            }}
        """

    def create_visual_element(self, color: str) -> str:
        return f'<div class="color-swatch" style="background-color: {color};"></div>'


if __name__ == '__main__':
    import folium

    discrete_color_mapping = DiscreteColorMapping(
        mapping={"Category A": "#ff0000", "Category B": "#00ff00", "Category C": "#0000ff"},
        default_output="#AAAAAA"
    )
    discrete_color_legend = DiscreteColorLegend(
        mapping=discrete_color_mapping,
        title="Discrete Color Categories",
        swatch_size=20,
        position={"top": 50, "right": 50},
        background_color=None,
    )

    m = folium.Map(location=[48.85, 2.35], zoom_start=10)
    m.add_child(discrete_color_legend)
    m.save("_tmp/map.html")
