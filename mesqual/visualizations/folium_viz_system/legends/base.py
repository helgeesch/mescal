from abc import ABC, abstractmethod
from typing import Generic, Any

from branca.element import MacroElement, Template

from mesqual.typevars import ValueMappingType


class BaseLegend(MacroElement, ABC, Generic[ValueMappingType]):
    """Base class for all mapping legends with common functionality"""

    _base_template = Template("""
    {% macro header(this, kwargs) %}
        <style>
            #{{ this.get_name() }} {
                position: fixed;
                z-index: 1000;
                background: {{ this.background_color }};
                padding: {{ this.padding }}px;
                border-radius: {{ this.border_radius }}px;
                box-shadow: 0 0 5px rgba(0,0,0,0.2);
                font-family: {{ this.font_family }};
                {% for key, value in this.position.items() %}
                {{ key }}: {{ value }};
                {% endfor %}
            }
            #{{ this.get_name() }} .legend-title {
                margin-bottom: {{ this.title_margin_bottom }}px;
                font-size: {{ this.title_font_size }}px;
                font-weight: bold;
                color: {{ this.title_color }};
            }
            #{{ this.get_name() }} .legend-content {
                position: relative;
                width: {{ this.width }}px;
            }
            {{ this.additional_styles() }}
        </style>
    {% endmacro %}

    {% macro html(this, kwargs) %}
        <div id="{{ this.get_name() }}">
            {% if this.title %}
            <div class="legend-title">{{ this.title }}</div>
            {% endif %}
            <div class="legend-content">
                {{ this.render_content() }}
            </div>
        </div>
    {% endmacro %}
    """)

    def __init__(
            self,
            mapping: ValueMappingType,
            title: str = "",
            background_color: str = "white",
            title_color: str = "#333333",
            title_font_size: int = 14,
            title_margin_bottom: int = 8,
            font_family: str = "Arial, sans-serif",
            width: int = 200,
            padding: int = 10,
            border_radius: int = 5,
            position: dict[str, int | float | str] | None = None
    ):
        super().__init__()
        self._name = f"{self.__class__.__name__}_{id(self)}"
        self.mapping = mapping
        self.title = title
        self.background_color = background_color
        self.title_color = title_color
        self.title_font_size = title_font_size
        self.title_margin_bottom = title_margin_bottom
        self.font_family = font_family
        self.width = width
        self.padding = padding
        self.border_radius = border_radius
        self.position = self._process_position(position or {"bottom": "20px", "right": "20px"})

        self._template = self._base_template

    def __call__(self, value) -> Any:
        return self.mapping(value)

    def _process_position(self, position: dict[str, int | float | str]) -> dict[str, str]:
        return {
            k: f"{v}px" if isinstance(v, (int, float)) else str(v)
            for k, v in position.items()
        }

    @abstractmethod
    def additional_styles(self) -> str:
        """Return additional CSS styles specific to this legend type"""
        pass

    @abstractmethod
    def render_content(self) -> str:
        """Return the HTML content for the legend"""
        pass

    def _format_value(self, value: float | int | str) -> str:
        """Format a value for display"""
        if isinstance(value, (int, float)):
            if isinstance(value, float) and value % 1 == 0:
                return f"{int(value)}"
            return f"{value:.2f}".rstrip('0').rstrip('.')
        return str(value)
