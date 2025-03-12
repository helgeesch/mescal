from dataclasses import dataclass, field

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio


class ConstantsIterable:
    @classmethod
    def items(cls):
        for attr_name in dir(cls):
            if not attr_name.startswith('__') and not callable(getattr(cls, attr_name)):
                yield attr_name, getattr(cls, attr_name)

    @classmethod
    def values(cls):
        for attr_name in dir(cls):
            if not attr_name.startswith('__') and not callable(getattr(cls, attr_name)):
                yield getattr(cls, attr_name)

    @classmethod
    def keys(cls):
        for attr_name in dir(cls):
            if not attr_name.startswith('__') and not callable(getattr(cls, attr_name)):
                yield attr_name


class PrimaryColors(ConstantsIterable):
    mint = '#00b894'
    cyan = '#00cec9'
    blue = '#0984e3'
    red = '#d63031'
    pink = '#e84393'
    green_light = '#badc58'
    green_bold = '#6ab04c'
    orange_light = '#fdcb6e'
    orange_bold = '#e17055'
    purple_light = '#a29bfe'
    purple_bold = '#6c5ce7'


class SequentialColors(ConstantsIterable):
    mint_blue_red = ['#00b894', '#0984e3', '#d63031']
    blue_cyan_pink = ['#0984e3', '#00cec9', '#e84393']
    shades_of_mint = ['#e6fff7', '#55efc4', '#00b894', '#009677', '#006b54']
    shades_of_cyan = ['#e6ffff', '#8ee8e7', '#00cec9', '#00a29a', '#00756e']
    shades_of_blue = ['#e6f4ff', '#74b9ff', '#0984e3', '#0063b1', '#004680']
    shades_of_red = ['#ffe6e6', '#ff7675', '#d63031', '#b02525', '#801b1b']
    shades_of_pink = ['#ffe6f3', '#fd79a8', '#e84393', '#c13584', '#962264']
    default = mint_blue_red


class DivergingColors(ConstantsIterable):
    blue_mint = SequentialColors.shades_of_blue[::-1] + SequentialColors.shades_of_mint
    red_mint = SequentialColors.shades_of_red[::-1] + SequentialColors.shades_of_mint
    pink_blue = SequentialColors.shades_of_pink[::-1] + SequentialColors.shades_of_blue
    pink_cyan = SequentialColors.shades_of_pink[::-1] + SequentialColors.shades_of_cyan
    default = blue_mint


class CyclicalColors(ConstantsIterable):
    default = None


class QualitativeColors(ConstantsIterable):
    default = [
        PrimaryColors.blue,
        PrimaryColors.mint,
        PrimaryColors.cyan,
        PrimaryColors.red,
        PrimaryColors.pink,
        PrimaryColors.green_light,
        PrimaryColors.green_bold,
        PrimaryColors.orange_light,
        PrimaryColors.orange_bold,
        PrimaryColors.purple_light,
        PrimaryColors.purple_bold,
    ]


class ColorPalette:
    primary = PrimaryColors
    sequential = SequentialColors
    diverging = DivergingColors
    cyclical = CyclicalColors
    qualitative = QualitativeColors


colors = ColorPalette


@dataclass
class PlotlyTheme:
    default_colorway: list[str] = field(default_factory=lambda: QualitativeColors.default)
    background_color: str = '#F2F2F2'
    paper_color: str = '#ffffff'
    watermark_text: str = None
    watermark_position: tuple[float, float] = (0.99, 0.01)
    watermark_opacity: float = 0.1
    font: dict[str, str] = field(default_factory=lambda: dict(family="Arial", size=12))

    def apply(self) -> None:
        template = go.layout.Template()

        template.layout.plot_bgcolor = self.background_color
        template.layout.paper_bgcolor = self.paper_color
        if self.font:
            template.layout.font = self.font
        template.layout.title = dict(x=0.5)

        template.layout.colorway = self.default_colorway

        if self.watermark_text:
            template.layout.annotations = [
                dict(
                    name='watermark',
                    text=self.watermark_text,
                    xref="paper",
                    yref="paper",
                    x=self.watermark_position[0],
                    y=self.watermark_position[1],
                    showarrow=False,
                    font=dict(size=50, color="black"),
                    opacity=self.watermark_opacity,
                    textangle=0,
                )
            ]

        pio.templates["custom"] = template
        pio.templates.default = "plotly+custom"


if __name__ == "__main__":
    theme = PlotlyTheme(watermark_text="mescal")

    theme.apply()

    # Example plot
    x = np.linspace(0, 10, 100)
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=x, y=np.sin(x), name="Sin"))
    fig.add_trace(go.Scatter(x=x, y=np.cos(x), name="Cos"))
    fig.update_layout(title='Sine and Cosine in Template Style')

    fig.show(renderer='browser')
