from abc import abstractmethod
from typing import Generic

from mesqual.typevars import DiscreteMappingType
from mesqual.visualizations.folium_viz_system.legends.base import BaseLegend


class DiscreteLegendBase(BaseLegend[DiscreteMappingType], Generic[DiscreteMappingType]):
    """Base class for discrete mapping legends with two-column layout"""

    def __init__(
            self,
            mapping: DiscreteMappingType,
            visual_column_width: int = 60,
            column_spacing: int = 10,
            row_spacing: int = 8,
            label_font_size: int = 12,
            **kwargs
    ):
        super().__init__(mapping, **kwargs)
        self.visual_column_width = visual_column_width
        self.column_spacing = column_spacing
        self.row_spacing = row_spacing
        self.label_font_size = label_font_size

    def additional_styles(self) -> str:
        id_selector = f"#{self.get_name()}"
        return f"""
            {id_selector} .discrete-item {{
                display: flex;
                align-items: center;
                margin-bottom: {self.row_spacing}px;
            }}
            {id_selector} .visual-column {{
                width: {self.visual_column_width}px;
                margin-right: {self.column_spacing}px;
                display: flex;
                align-items: center;
                justify-content: center;
            }}
            {id_selector} .label-column {{
                flex: 1;
                font-size: {self.label_font_size}px;
                color: {self.title_color};
            }}
            {self.specific_visual_styles()}
        """

    @abstractmethod
    def specific_visual_styles(self) -> str:
        """Return CSS styles specific to the visual representation"""
        pass

    @abstractmethod
    def create_visual_element(self, value: any) -> str:
        """Create the visual representation for a given value"""
        pass

    def render_content(self) -> str:
        items = []

        # Render mapped items
        for key, value in sorted(self.mapping.mapping.items()):
            visual = self.create_visual_element(value)
            items.append(f"""
                <div class="discrete-item">
                    <div class="visual-column">{visual}</div>
                    <div class="label-column">{self._format_value(key)}</div>
                </div>
            """)

        if self.mapping.default_output is not None:
            visual = self.create_visual_element(self.mapping.default_output)
            items.append(f"""
                <div class="discrete-item">
                    <div class="visual-column">{visual}</div>
                    <div class="label-column">Other</div>
                </div>
            """)

        return ''.join(items)
