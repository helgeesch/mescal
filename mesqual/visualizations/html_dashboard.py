import os
from typing import List, Union, Dict, Tuple, Optional, TYPE_CHECKING
from collections import OrderedDict

import plotly.graph_objects as go

from mesqual.utils.logging import get_logger

if TYPE_CHECKING:
    import folium
    from mesqual.visualizations.html_table import HTMLTable

logger = get_logger(__name__)


class HTMLDashboardElement:
    """A wrapper for dashboard elements that can be HTML strings or Plotly figures.

    This class serves as a container for individual dashboard components, storing
    the element content along with display properties like height and a unique name.

    Args:
        element: The dashboard element content, either a Plotly figure or HTML string.
        height: CSS height specification for the element. Defaults to '100%'.
        name: Unique identifier for the element. If None, auto-generates using object id.
        tab: Optional tab path. A string places the element in a top-level tab.
            A tuple of two strings places it in a nested sub-tab, e.g.
            ``("Market Results", "Volumes")``. If None, the element is ungrouped
            (rendered at the top level in scroll mode, or placed in a default tab
            in tabbed mode).

    Attributes:
        element: The stored dashboard element (figure or HTML string).
        height: The CSS height specification.
        name: The unique identifier for this element.
        tab: The tab path, normalised to a tuple or None.
    """
    def __init__(
            self,
            element: Union[go.Figure, str],
            height: str = '100%',
            name: str = None,
            tab: Union[str, Tuple[str, str], None] = None,
    ):
        self.element = element
        self.height = height
        self.name = name or str(id(self))
        if name is None:
            logger.info(f'No name passed for {type(self).__name__}. Automatically generated name: {self.name}')

        # Normalise tab to tuple or None
        if isinstance(tab, str):
            self.tab = (tab,)
        elif isinstance(tab, tuple):
            if len(tab) not in (1, 2):
                raise ValueError(f"tab tuple must have 1 or 2 elements, got {len(tab)}")
            self.tab = tab
        else:
            self.tab = None


class HTMLDashboard:
    """A dashboard builder for creating HTML reports with multiple visualizations.

    This class provides a flexible way to combine Plotly figures, Folium maps,
    HTML tables, custom HTML content, and section dividers into a single HTML
    dashboard file. Elements are stored with unique names and can be ordered
    when saving the final dashboard.

    Args:
        name: The dashboard title. Defaults to 'HTML Dashboard'.
        font_family: CSS font family specification for the dashboard.
            Defaults to "Arial, sans-serif".

    Attributes:
        name: The dashboard title.
        content: Dictionary mapping element names to HTMLDashboardElement objects.
        font_family: The CSS font family specification.

    Example:

        >>> import plotly.express as px
        >>> dashboard = HTMLDashboard(name="My Analysis")
        >>> fig = px.scatter(px.data.iris(), x="sepal_width", y="sepal_length")
        >>> dashboard.add_plotly_figure(fig, name="iris_scatter")
        >>> dashboard.save("analysis.html")
    """
    def __init__(self, name: str = None, font_family: str = "Arial, sans-serif"):
        self.name = name if name else 'HTML Dashboard'
        self.content: Dict[str, HTMLDashboardElement] = dict()
        self.font_family = font_family

    def add_plotly_figure(
            self,
            fig: go.Figure,
            height: str = '100%',
            name: str = None,
            tab: Union[str, Tuple[str, str], None] = None,
    ):
        """Add a Plotly figure to the dashboard.

        Args:
            fig: The Plotly figure to add.
            height: CSS height specification for the figure. Defaults to '100%'.
            name: Unique identifier for the figure. If None, auto-generates.
            tab: Optional tab path (see :class:`HTMLDashboardElement`).

        Example:

            >>> import plotly.express as px
            >>> fig = px.bar(x=["A", "B", "C"], y=[1, 3, 2])
            >>> dashboard.add_plotly_figure(fig, height="400px", name="my_bar_chart")
        """
        element = HTMLDashboardElement(fig, height, name, tab=tab)
        self.content[element.name] = element

    def add_html(self, html_string: str, name: str = None, tab: Union[str, Tuple[str, str], None] = None):
        """Add custom HTML content to the dashboard.

        Args:
            html_string: The HTML content to add.
            name: Unique identifier for the HTML content. If None, auto-generates.
            tab: Optional tab path (see :class:`HTMLDashboardElement`).

        Example:

            >>> html = "<div><h2>Custom Section</h2><p>Some content here.</p></div>"
            >>> dashboard.add_html(html, name="custom_section")
        """
        element = HTMLDashboardElement(html_string, name=name, tab=tab)
        self.content[element.name] = element

    def add_folium_map(
            self,
            folium_map: 'folium.Map',
            name: str = None,
            tab: Union[str, Tuple[str, str], None] = None,
    ):
        """Add a Folium map to the dashboard.

        Args:
            folium_map: The Folium map object to add.
            name: Unique identifier for the map. If None, auto-generates
                as "folium_map_{index}".
            tab: Optional tab path (see :class:`HTMLDashboardElement`).

        Returns:
            str: The name assigned to the map element.

        Example:

            >>> import folium
            >>> m = folium.Map(location=[45.5236, -122.6750], zoom_start=13)
            >>> map_name = dashboard.add_folium_map(m, name="portland_map")
        """
        map_html = folium_map._repr_html_()

        if name is None:
            name = f"folium_map_{len([k for k in self.content.keys() if 'folium_map' in k])}"

        wrapped_map_html = f'<div>{map_html}</div>'

        self.add_html(wrapped_map_html, name=name, tab=tab)

        return name

    def add_table(
            self,
            table: 'HTMLTable',
            name: str = None,
            include_dependencies: bool = True,
            tab: Union[str, Tuple[str, str], None] = None,
    ) -> str:
        """Add an HTML table to the dashboard.

        Args:
            table: The HTMLTable object to add.
            name: Unique identifier for the table. If None, derives from table title
                or uses table_id.
            include_dependencies: Whether to include CSS/JS dependencies in the
                table HTML. Defaults to True.
            tab: Optional tab path (see :class:`HTMLDashboardElement`).

        Returns:
            str: The name assigned to the table element.

        Raises:
            ValueError: If the table cannot be converted to HTML.

        Example:

            >>> from mesqual.visualizations.html_table import HTMLTable
            >>> import pandas as pd
            >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
            >>> table = HTMLTable(df, title="Sample Data")
            >>> table_name = dashboard.add_table(table, name="sample_table")
        """
        try:
            table_html = table.get_html(include_dependencies=include_dependencies)

            if name is None:
                name = f"table_{table.title.lower().replace(' ', '_')}" if table.title else table.table_id

            self.add_html(table_html, name=name, tab=tab)
            return name

        except Exception as e:
            raise ValueError(f"Failed to add table to dashboard: {str(e)}") from e

    def add_section_divider(
            self,
            title: str,
            subtitle: str = None,
            name: str = None,
            tab: Union[str, Tuple[str, str], None] = None,
            background_color: str = "#f9f9f9",
            title_color: str = "#333",
            subtitle_color: str = "#666",
            padding: str = "20px",
            margin: str = "20px 0",
            text_align: str = "center",
            title_font_size: str = "24px",
            subtitle_font_size: str = "16px",
            border_radius: str = "0px",
            border: str = "none",
            **kwargs
    ) -> str:
        """Add a styled section divider with title and optional subtitle.

        Creates a formatted section header that can be used to organize dashboard
        content into logical groups. Supports extensive CSS customization through
        parameters and keyword arguments.

        Args:
            title: The main section title.
            subtitle: Optional subtitle text.
            name: Unique identifier for the divider. If None, derives from title.
            tab: Optional tab path (see :class:`HTMLDashboardElement`).
            background_color: CSS background color. Defaults to "#f9f9f9".
            title_color: CSS color for the title text. Defaults to "#333".
            subtitle_color: CSS color for the subtitle text. Defaults to "#666".
            padding: CSS padding specification. Defaults to "20px".
            margin: CSS margin specification. Defaults to "20px 0".
            text_align: CSS text alignment. Defaults to "center".
            title_font_size: CSS font size for title. Defaults to "24px".
            subtitle_font_size: CSS font size for subtitle. Defaults to "16px".
            border_radius: CSS border radius. Defaults to "0px".
            border: CSS border specification. Defaults to "none".
            **kwargs: Additional CSS properties. Underscores in keys are converted
                to camelCase (e.g., box_shadow becomes boxShadow).

        Returns:
            str: The name assigned to the section divider element.

        Example:

            >>> divider_name = dashboard.add_section_divider(
            ...     title="Data Analysis Results",
            ...     subtitle="Generated on 2024-01-01",
            ...     background_color="#e3f2fd",
            ...     border="1px solid #2196f3"
            ... )
        """
        base_style = f"background-color: {background_color}; padding: {padding}; margin: {margin}; text-align: {text_align}; border-radius: {border_radius}; border: {border};"

        for key, value in kwargs.items():
            css_key = ''.join(word.capitalize() if i > 0 else word for i, word in enumerate(key.split('_')))
            base_style += f" {css_key}: {value};"

        html = f'<div style="{base_style}">\n'
        html += f'    <h2 style="color: {title_color}; font-size: {title_font_size};">{title}</h2>\n'

        if subtitle:
            html += f'    <p style="color: {subtitle_color}; font-size: {subtitle_font_size};">{subtitle}</p>\n'

        html += '</div>'

        if name is None:
            name = f"section_{title.lower().replace(' ', '_')}"

        self.add_html(html, name=name, tab=tab)

        return name

    # -- internal helpers ------------------------------------------------

    def _has_tabs(self) -> bool:
        return any(el.tab is not None for el in self.content.values())

    def _resolve_content_order(self, content_order):
        if content_order is None:
            return list(self.content.keys())
        unrecognized = [k for k in content_order if k not in self.content.keys()]
        if unrecognized:
            raise KeyError(f'Unrecognized content names: {unrecognized}. Allowed: {self.content.keys()}')
        return content_order

    def _element_to_html(self, element: HTMLDashboardElement, plotly_js_included: bool) -> str:
        if isinstance(element.element, go.Figure):
            return element.element.to_html(
                include_plotlyjs=not plotly_js_included,
                full_html=False,
                default_height=element.height,
            )
        elif isinstance(element.element, str):
            return element.element
        else:
            raise TypeError(f'Unexpected element type: {type(element.element)}')

    DEFAULT_TAB = '.'

    def _build_tab_structure(self, content_order):
        """Return an OrderedDict representing the tab tree.

        Structure::

            {
                "Market Results": {          # top-level tab
                    "_elements": [...],      # elements directly in this tab
                    "Volumes": {             # sub-tab
                        "_elements": [...],
                    },
                },
                ...
            }
        """
        tabs: OrderedDict = OrderedDict()

        for key in content_order:
            el = self.content[key]
            path = el.tab if el.tab is not None else (self.DEFAULT_TAB,)

            top = path[0]
            if top not in tabs:
                tabs[top] = OrderedDict(_elements=[])

            if len(path) == 2:
                sub = path[1]
                if sub not in tabs[top]:
                    tabs[top][sub] = OrderedDict(_elements=[])
                tabs[top][sub]['_elements'].append(key)
            else:
                tabs[top]['_elements'].append(key)

        return tabs

    @staticmethod
    def _tab_css() -> str:
        return """
        html, body { height: 100%; margin: 0; }
        #tabgroup_top { display: flex; flex-direction: column; height: 100vh; }
        .dashboard-tabs { flex: 0 0 auto; display: flex; flex-wrap: wrap; border-bottom: 2px solid #dee2e6; margin: 0; padding: 0; gap: 2px; }
        .dashboard-tabs button {
            padding: 10px 20px; border: 1px solid transparent; border-bottom: none;
            background: #f8f9fa; cursor: pointer; font-size: 14px; color: #495057;
            border-radius: 6px 6px 0 0; transition: background 0.15s, color 0.15s;
        }
        .dashboard-tabs button:hover { background: #e9ecef; }
        .dashboard-tabs button.active { background: #fff; color: #212529; border-color: #dee2e6; font-weight: 600; position: relative; }
        .dashboard-tabs button.active::after { content: ''; position: absolute; bottom: -2px; left: 0; right: 0; height: 2px; background: #fff; }
        .dashboard-tab-content { display: none; flex: 1 1 0; min-height: 0; overflow: auto; padding: 0; }
        .dashboard-tab-content.active { display: flex; flex-direction: column; }
        .dashboard-subtabs { flex: 0 0 auto; display: flex; flex-wrap: wrap; border-bottom: 1px solid #e9ecef; margin: 0; padding: 0; gap: 2px; }
        .dashboard-subtabs button {
            padding: 6px 16px; border: 1px solid transparent; border-bottom: none;
            background: #f8f9fa; cursor: pointer; font-size: 13px; color: #6c757d;
            border-radius: 4px 4px 0 0; transition: background 0.15s, color 0.15s;
        }
        .dashboard-subtabs button:hover { background: #e9ecef; }
        .dashboard-subtabs button.active { background: #fff; color: #212529; border-color: #e9ecef; font-weight: 600; position: relative; }
        .dashboard-subtabs button.active::after { content: ''; position: absolute; bottom: -1px; left: 0; right: 0; height: 1px; background: #fff; }
        .dashboard-subtab-content { display: none; flex: 1 1 0; min-height: 0; overflow: auto; padding: 0; }
        .dashboard-subtabgroup { display: flex; flex-direction: column; flex: 1 1 0; min-height: 0; }
        .dashboard-subtab-content.active { display: flex; flex-direction: column; }
        .dashboard-tab-content .js-plotly-plot,
        .dashboard-subtab-content .js-plotly-plot { width: 100% !important; flex: 1 1 0; min-height: 0; }
        .dashboard-tab-content .plotly-graph-div,
        .dashboard-subtab-content .plotly-graph-div { height: 100% !important; }
        """

    @staticmethod
    def _tab_js() -> str:
        return """
        function switchTab(groupId, tabId) {
            var group = document.getElementById(groupId);
            group.querySelectorAll(':scope > .dashboard-tab-content, :scope > .dashboard-subtab-content').forEach(function(el) {
                el.classList.remove('active');
            });
            var tabs = group.querySelector('.dashboard-tabs') || group.querySelector('.dashboard-subtabs');
            tabs.querySelectorAll('button').forEach(function(btn) { btn.classList.remove('active'); });
            document.getElementById(tabId).classList.add('active');
            document.querySelector('[onclick*=\"\\'' + tabId + '\\'\"]').classList.add('active');
            // Plotly figures rendered inside hidden tabs have zero dimensions;
            // resize them once the tab becomes visible.
            var target = document.getElementById(tabId);
            var plots = target.querySelectorAll('.plotly-graph-div');
            plots.forEach(function(p) { if (window.Plotly) Plotly.Plots.resize(p); });
        }
        """

    def _render_tabbed(self, content_order) -> str:
        plotly_js_included = False
        tab_tree = self._build_tab_structure(content_order)
        parts = []
        counter = 0

        def next_id(prefix='tab'):
            nonlocal counter
            counter += 1
            return f'{prefix}_{counter}'

        # Top-level tab bar
        group_id = 'tabgroup_top'
        parts.append(f'<div id="{group_id}">')
        parts.append('<div class="dashboard-tabs">')

        tab_ids = {}
        first_top = True
        for tab_label in tab_tree:
            tid = next_id('tab')
            tab_ids[tab_label] = tid
            active = ' active' if first_top else ''
            parts.append(
                f"<button class=\"{active.strip()}\" onclick=\"switchTab('{group_id}', '{tid}')\">{tab_label}</button>"
            )
            first_top = False
        parts.append('</div>')

        # Top-level tab contents
        first_top = True
        for tab_label, tab_data in tab_tree.items():
            tid = tab_ids[tab_label]
            active = ' active' if first_top else ''
            parts.append(f'<div id="{tid}" class="dashboard-tab-content{active}">')

            # Direct elements in this tab
            sub_tabs = OrderedDict(
                (k, v) for k, v in tab_data.items() if k != '_elements'
            )

            if sub_tabs:
                # Render sub-tab bar
                sub_group_id = next_id('subtabgroup')
                parts.append(f'<div id="{sub_group_id}" class="dashboard-subtabgroup">')
                parts.append('<div class="dashboard-subtabs">')

                # If there are direct elements, they go in a "General" sub-tab
                direct_elements = tab_data['_elements']
                sub_tab_ids = {}
                first_sub = True

                if direct_elements:
                    stid = next_id('subtab')
                    sub_tab_ids['_direct'] = stid
                    active_s = ' active' if first_sub else ''
                    parts.append(
                        f"<button class=\"{active_s.strip()}\" onclick=\"switchTab('{sub_group_id}', '{stid}')\">{self.DEFAULT_TAB}</button>"
                    )
                    first_sub = False

                for sub_label in sub_tabs:
                    stid = next_id('subtab')
                    sub_tab_ids[sub_label] = stid
                    active_s = ' active' if first_sub else ''
                    parts.append(
                        f"<button class=\"{active_s.strip()}\" onclick=\"switchTab('{sub_group_id}', '{stid}')\">{sub_label}</button>"
                    )
                    first_sub = False

                parts.append('</div>')

                # Sub-tab contents
                first_sub = True
                if direct_elements:
                    stid = sub_tab_ids['_direct']
                    active_s = ' active' if first_sub else ''
                    parts.append(f'<div id="{stid}" class="dashboard-subtab-content{active_s}">')
                    for key in direct_elements:
                        el = self.content[key]
                        parts.append(self._element_to_html(el, plotly_js_included))
                        if isinstance(el.element, go.Figure):
                            plotly_js_included = True
                    parts.append('</div>')
                    first_sub = False

                for sub_label, sub_data in sub_tabs.items():
                    stid = sub_tab_ids[sub_label]
                    active_s = ' active' if first_sub else ''
                    parts.append(f'<div id="{stid}" class="dashboard-subtab-content{active_s}">')
                    for key in sub_data['_elements']:
                        el = self.content[key]
                        parts.append(self._element_to_html(el, plotly_js_included))
                        if isinstance(el.element, go.Figure):
                            plotly_js_included = True
                    parts.append('</div>')
                    first_sub = False

                parts.append('</div>')  # close sub_group_id
            else:
                # No sub-tabs — just render direct elements
                for key in tab_data['_elements']:
                    el = self.content[key]
                    parts.append(self._element_to_html(el, plotly_js_included))
                    if isinstance(el.element, go.Figure):
                        plotly_js_included = True

            parts.append('</div>')  # close tab content
            first_top = False

        parts.append('</div>')  # close tabgroup_top
        return '\n'.join(parts)

    # -- public API ----------------------------------------------------

    def save(self, save_to_path, content_order=None):
        """Save the dashboard as an HTML file.

        If any element has a ``tab`` set, the dashboard is rendered with a tabbed
        layout. Otherwise it falls back to the classic scrollable layout.

        Args:
            save_to_path: File path where the HTML dashboard will be saved.
            content_order: Optional list specifying the order of elements in the
                dashboard. If None, uses the order elements were added. Must
                contain only valid element names.

        Raises:
            KeyError: If content_order contains names not found in the dashboard.
            TypeError: If an element has an unexpected type (internal error).

        Example:

            >>> dashboard.save("my_dashboard.html")
            >>> # Custom ordering
            >>> dashboard.save("ordered_dashboard.html",
            ...               content_order=["intro_section", "chart1", "table1"])
        """
        dir_name = os.path.dirname(save_to_path)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name)

        content_order = self._resolve_content_order(content_order)
        use_tabs = self._has_tabs()

        with open(save_to_path, 'w', encoding='utf-8') as f:
            f.write("<html><head>\n")
            f.write("<meta charset='UTF-8'>\n")
            f.write(f"<title>{self.name}</title>\n")
            f.write(f"<style>\n  body, * {{ font-family: {self.font_family}; }}\n")
            if use_tabs:
                f.write(self._tab_css())
            f.write("</style>\n")
            f.write("</head><body>\n")

            if use_tabs:
                f.write(self._render_tabbed(content_order))
                f.write(f"\n<script>\n{self._tab_js()}\n</script>\n")
            else:
                plotly_js_included = False
                for key in content_order:
                    el = self.content[key]
                    f.write(self._element_to_html(el, plotly_js_included) + "\n")
                    if isinstance(el.element, go.Figure):
                        plotly_js_included = True

            f.write("</body></html>\n")

    def show(self, width: str = "100%", height: str = "600"):
        """Display the dashboard inline in a Jupyter notebook.

        Creates a temporary HTML file and displays it using an IPython IFrame.
        This method is designed for use within Jupyter notebooks to provide
        inline dashboard previews.

        Args:
            width: CSS width specification for the iframe. Defaults to "100%".
            height: CSS height specification for the iframe. Defaults to "600".

        Note:
            This method requires IPython and is intended for Jupyter notebook use.
            The temporary file is created in the system temp directory.

        Example:

            >>> # In a Jupyter notebook cell
            >>> dashboard.show(width="100%", height="800px")
        """
        import tempfile
        from pathlib import Path
        from IPython.display import IFrame, display

        tmp_dir = Path(tempfile.mkdtemp())
        html_path = tmp_dir / "dashboard.html"
        self.save(html_path)
        display(IFrame(src=str(html_path.resolve()), width=width, height=height))


if __name__ == '__main__':
    from mesqual.visualizations.html_table import HTMLTable
    import plotly.express as px
    import plotly.graph_objects as go

    # Load sample data
    data = px.data.iris()

    # ── Classic scrollable dashboard (no tabs) ──────────────────────────
    dashboard = HTMLDashboard(
        name='MESQUAL Visualization Dashboard (scroll)',
        font_family="'Segoe UI', Tahoma, Geneva, Verdana, sans-serif"
    )

    dashboard.add_section_divider(
        title='MESQUAL Dashboard Example',
        subtitle='Demonstrating multiple visualization types',
        background_color="#2c3e50",
        title_color="white",
        subtitle_color="#ecf0f1",
        border_radius="8px"
    )

    scatter_fig = px.scatter(data, x="sepal_width", y="sepal_length", color="species",
                             title="Sepal Dimensions by Species")
    dashboard.add_plotly_figure(scatter_fig, height="450px", name="sepal_scatter")

    box_fig = px.box(data, x="species", y="petal_length", title="Petal Length Distribution")
    dashboard.add_plotly_figure(box_fig, height="400px", name="petal_boxplot")

    dashboard.save('_tmp/figure_dashboard_scroll.html')
    print("Scroll dashboard saved to '_tmp/figure_dashboard_scroll.html'")

    # ── Tabbed dashboard ────────────────────────────────────────────────
    tabbed = HTMLDashboard(
        name='MESQUAL Visualization Dashboard (tabs)',
        font_family="'Segoe UI', Tahoma, Geneva, Verdana, sans-serif"
    )

    # Elements without tab → go to default "." tab
    tabbed.add_section_divider(
        title='MESQUAL Dashboard Example',
        subtitle='Demonstrating the tabbed layout',
        background_color="#2c3e50",
        title_color="white",
        subtitle_color="#ecf0f1",
        border_radius="8px"
    )

    # Top-level tab "Charts"
    tabbed.add_plotly_figure(scatter_fig, height="450px", name="sepal_scatter",
                             tab="Charts")
    tabbed.add_plotly_figure(box_fig, height="400px", name="petal_boxplot",
                             tab="Charts")

    # Nested sub-tab ("Charts", "Custom")
    custom_fig = go.Figure()
    custom_fig.add_scatter(
        x=data["sepal_width"], y=data["petal_length"], mode='markers',
        marker=dict(size=8, color='lightblue', line=dict(width=1, color='navy')),
        name='All Species',
    )
    custom_fig.update_layout(title="Custom Scatter Plot with Styling",
                             xaxis_title="Sepal Width (cm)",
                             yaxis_title="Petal Length (cm)")
    tabbed.add_plotly_figure(custom_fig, name="custom_scatter",
                             tab=("Charts", "Custom"))

    # Another top-level tab "Data"
    sample_data = data.head(10)
    table = HTMLTable(sample_data, title="Iris Dataset Sample")
    tabbed.add_table(table, name="iris_sample_table", tab="Data")

    custom_html = """
    <div style="background-color: #f8f9fa; padding: 20px; border-left: 4px solid #007bff;">
        <h3 style="color: #007bff; margin-top: 0;">Analysis Summary</h3>
        <ul>
            <li><strong>Dataset:</strong> Iris flower measurements</li>
            <li><strong>Records:</strong> 150 samples</li>
            <li><strong>Species:</strong> Setosa, Versicolor, Virginica</li>
            <li><strong>Variables:</strong> Sepal/Petal length and width</li>
        </ul>
    </div>
    """
    tabbed.add_html(custom_html, name="analysis_summary", tab="Data")

    tabbed.save('_tmp/figure_dashboard_tabs.html')
    print("Tabbed dashboard saved to '_tmp/figure_dashboard_tabs.html'")
    print(f"Dashboard contains {len(tabbed.content)} elements:")
    for name, element in tabbed.content.items():
        element_type = "Plotly Figure" if isinstance(element.element, go.Figure) else "HTML Content"
        tab_info = f" (tab: {element.tab})" if element.tab else ""
        print(f"  - {name}: {element_type}{tab_info}")
