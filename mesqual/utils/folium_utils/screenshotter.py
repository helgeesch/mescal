from pathlib import Path
from dataclasses import dataclass
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from PIL import Image
import io
import time


@dataclass
class ScreenConfig:
    """Configuration for the virtual browser viewport.

    Attributes:
        width: Viewport width in CSS pixels.
        height: Viewport height in CSS pixels.
        device_pixel_ratio: Scale factor for high-DPI rendering.
            A value of 2.0 produces 2x resolution output (e.g., 1920x1080 viewport -> 3840x2160 pixels).
    """
    width: int = 1920
    height: int = 1080
    device_pixel_ratio: float = 1.0


@dataclass
class FrameConfig:
    """Configuration for the screenshot crop frame.

    The frame is centered on the screen by default. Use offsets to shift the crop area.

    Attributes:
        width: Frame width in CSS pixels.
        height: Frame height in CSS pixels.
        offset_center_x: Horizontal offset from screen center. Positive values shift right.
        offset_center_y: Vertical offset from screen center. Positive values shift down.
    """
    width: int
    height: int
    offset_center_x: int = 0
    offset_center_y: int = 0

    def to_crop_box(self, image_width: int, image_height: int, dpr: float) -> tuple[int, int, int, int]:
        """Convert frame config to PIL crop box coordinates.

        Args:
            image_width: Actual screenshot width in pixels.
            image_height: Actual screenshot height in pixels.
            dpr: Device pixel ratio for scaling.

        Returns:
            Tuple of (left, top, right, bottom) pixel coordinates.
        """
        center_x = image_width // 2 + int(self.offset_center_x * dpr)
        center_y = image_height // 2 + int(self.offset_center_y * dpr)
        scaled_width = int(self.width * dpr)
        scaled_height = int(self.height * dpr)
        left = center_x - scaled_width // 2
        top = center_y - scaled_height // 2
        right = left + scaled_width
        bottom = top + scaled_height
        return (left, top, right, bottom)


@dataclass
class MapViewConfig:
    """Configuration for the Leaflet map view.

    Attributes:
        center_lat: Latitude of map center.
        center_lng: Longitude of map center.
        zoom: Map zoom level. Supports fractional values (e.g., 4.5).
    """
    center_lat: float | None = None
    center_lng: float | None = None
    zoom: float | None = None


@dataclass
class LegendInfo:
    """Information about a detected legend element.

    Attributes:
        element_id: DOM element ID of the legend.
        title: Legend title text, if present.
        width: Legend width in CSS pixels.
        height: Legend height in CSS pixels.
    """
    element_id: str
    title: str | None
    width: int
    height: int


class FoliumScreenshotter:
    """Automated screenshot capture for Folium HTML maps.

    Uses headless Chrome via Selenium to render Folium maps and capture screenshots
    of map layers and legends. Supports high-DPI rendering, custom crop frames,
    and automatic detection of base layers and legend elements.

    Args:
        html_path: Path to the Folium-generated HTML file.
        screen_config: Virtual browser viewport configuration.
        frame_config: Screenshot crop frame configuration. If None, captures full viewport.
        map_view_config: Leaflet map view settings (center, zoom).

    Example:

        >>> screenshotter = FoliumScreenshotter(
        ...     "map.html",
        ...     screen_config=ScreenConfig(1920, 1080, 2.0),
        ...     frame_config=FrameConfig(800, 600),
        ...     map_view_config=MapViewConfig(52.52, 13.405, 10),
        ... )
        >>>
        >>> # Capture all base layers
        >>> screenshotter.capture_all_base_layers("output/layers")
        >>>
        >>> # Capture legends separately
        >>> screenshotter.capture_legends("output/legends")

    """

    def __init__(
            self,
            html_path: str | Path,
            screen_config: ScreenConfig | None = None,
            frame_config: FrameConfig | None = None,
            map_view_config: MapViewConfig | None = None
    ):
        self._html_path = Path(html_path).resolve()
        self._screen_config = screen_config or ScreenConfig()
        self._frame_config = frame_config
        self._map_view_config = map_view_config or MapViewConfig()
        self._driver: webdriver.Chrome | None = None

    def _start_browser(self) -> None:
        options = Options()
        options.add_argument("--headless=new")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument(f"--window-size={self._screen_config.width},{self._screen_config.height}")
        options.add_argument(f"--force-device-scale-factor={self._screen_config.device_pixel_ratio}")
        options.add_argument("--high-dpi-support=1")
        self._driver = webdriver.Chrome(options=options)

    def _stop_browser(self) -> None:
        if self._driver:
            self._driver.quit()
            self._driver = None

    def _load_map(self) -> None:
        self._driver.get(f"file://{self._html_path}")
        WebDriverWait(self._driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "leaflet-container"))
        )
        self._driver.execute_script("""
            let container = document.querySelector('.leaflet-container');
            container.style.position = 'absolute';
            container.style.top = '0';
            container.style.left = '0';
            container.style.width = '100vw';
            container.style.height = '100vh';
            container.style.margin = '0';
            container.style.padding = '0';
            document.body.style.margin = '0';
            document.body.style.padding = '0';
            document.body.style.overflow = 'hidden';
        """)
        time.sleep(1)

    def _get_map_object_name(self) -> str:
        return self._driver.execute_script("""
            for (let key in window) {
                if (window[key] instanceof L.Map) return key;
            }
            return null;
        """)

    def _set_view(self, map_name: str) -> None:
        cfg = self._map_view_config
        if cfg.center_lat is not None and cfg.center_lng is not None:
            zoom_expr = str(cfg.zoom) if cfg.zoom else f"{map_name}.getZoom()"
            self._driver.execute_script(
                f"{map_name}.setView([{cfg.center_lat}, {cfg.center_lng}], {zoom_expr});"
            )
        elif cfg.zoom is not None:
            self._driver.execute_script(f"{map_name}.setZoom({cfg.zoom});")
        self._driver.execute_script(f"{map_name}.invalidateSize();")
        time.sleep(1)

    def _get_base_layers(self) -> list[dict]:
        return self._driver.execute_script("""
            let result = [];
            document.querySelectorAll('.leaflet-control-layers-base label').forEach((label, idx) => {
                let input = label.querySelector('input');
                let name = label.textContent.trim();
                result.push({
                    name: name,
                    index: idx,
                    checked: input.checked
                });
            });
            return result;
        """)

    def _select_base_layer(self, index: int) -> None:
        self._driver.execute_script(f"""
            let inputs = document.querySelectorAll('.leaflet-control-layers-base input');
            if (inputs[{index}]) inputs[{index}].click();
        """)
        time.sleep(1)

    def _detect_legends(self) -> list[LegendInfo]:
        legends_data = self._driver.execute_script("""
            let legends = [];

            // Method 1: Find by ID pattern (*Legend_*)
            document.querySelectorAll('[id*="Legend_"]').forEach(el => {
                let titleEl = el.querySelector('.legend-title');
                let rect = el.getBoundingClientRect();
                legends.push({
                    id: el.id,
                    title: titleEl ? titleEl.textContent.trim() : null,
                    width: rect.width,
                    height: rect.height
                });
            });

            // Method 2: Find by structure (position:fixed + .legend-content)
            if (legends.length === 0) {
                document.querySelectorAll('div').forEach(el => {
                    let style = window.getComputedStyle(el);
                    if (style.position === 'fixed' && el.querySelector('.legend-content')) {
                        let titleEl = el.querySelector('.legend-title');
                        let rect = el.getBoundingClientRect();
                        legends.push({
                            id: el.id || `legend_${legends.length}`,
                            title: titleEl ? titleEl.textContent.trim() : null,
                            width: rect.width,
                            height: rect.height
                        });
                    }
                });
            }

            return legends;
        """)

        return [
            LegendInfo(
                element_id=l["id"],
                title=l["title"],
                width=int(l["width"]),
                height=int(l["height"])
            )
            for l in legends_data
        ]

    def _hide_layer_control(self) -> None:
        self._driver.execute_script("""
            let control = document.querySelector('.leaflet-control-layers');
            if (control) control.style.display = 'none';
        """)

    def _show_layer_control(self) -> None:
        self._driver.execute_script("""
            let control = document.querySelector('.leaflet-control-layers');
            if (control) control.style.display = '';
        """)

    def _hide_legends(self) -> None:
        self._driver.execute_script("""
            // By ID pattern
            document.querySelectorAll('[id*="Legend_"]').forEach(el => {
                el.style.display = 'none';
            });

            // By structure (position:fixed + .legend-content)
            document.querySelectorAll('div').forEach(el => {
                let style = window.getComputedStyle(el);
                if (style.position === 'fixed' && el.querySelector('.legend-content')) {
                    el.style.display = 'none';
                }
            });
        """)

    def _show_legends(self) -> None:
        self._driver.execute_script("""
            document.querySelectorAll('[id*="Legend_"]').forEach(el => {
                el.style.display = '';
            });

            document.querySelectorAll('div').forEach(el => {
                let style = window.getComputedStyle(el);
                if (style.position === 'fixed' && el.querySelector('.legend-content')) {
                    el.style.display = '';
                }
            });
        """)

    def _hide_ui_elements(self) -> None:
        self._hide_layer_control()
        self._hide_legends()

    def _show_ui_elements(self) -> None:
        self._show_layer_control()
        self._show_legends()

    def _take_screenshot(self, output_path: Path) -> None:
        self._hide_ui_elements()

        png_bytes = self._driver.get_screenshot_as_png()
        image = Image.open(io.BytesIO(png_bytes))

        if self._frame_config:
            crop_box = self._frame_config.to_crop_box(
                image.width,
                image.height,
                self._screen_config.device_pixel_ratio
            )
            crop_box = (
                max(0, crop_box[0]),
                max(0, crop_box[1]),
                min(image.width, crop_box[2]),
                min(image.height, crop_box[3])
            )
            image = image.crop(crop_box)

        image.save(output_path)

        self._show_ui_elements()

    def _take_element_screenshot(self, element_id: str, output_path: Path) -> None:
        self._hide_layer_control()

        self._driver.execute_script("""
            let targetId = arguments[0];
            document.querySelectorAll('[id*="Legend_"]').forEach(el => {
                if (el.id !== targetId) el.style.display = 'none';
            });
        """, element_id)

        dpr = self._screen_config.device_pixel_ratio
        rect = self._driver.execute_script("""
            let el = document.getElementById(arguments[0]);
            let rect = el.getBoundingClientRect();
            return {left: rect.left, top: rect.top, width: rect.width, height: rect.height};
        """, element_id)

        png_bytes = self._driver.get_screenshot_as_png()
        image = Image.open(io.BytesIO(png_bytes))

        crop_box = (
            int(rect["left"] * dpr),
            int(rect["top"] * dpr),
            int((rect["left"] + rect["width"]) * dpr),
            int((rect["top"] + rect["height"]) * dpr)
        )

        crop_box = (
            max(0, crop_box[0]),
            max(0, crop_box[1]),
            min(image.width, crop_box[2]),
            min(image.height, crop_box[3])
        )

        image = image.crop(crop_box)
        image.save(output_path)

        self._show_ui_elements()

    def _sanitize_filename(self, name: str) -> str:
        return "".join(c if c.isalnum() or c in "._- " else "_" for c in name).strip()

    def set_frame(
            self,
            width: int,
            height: int,
            offset_center_x: int = 0,
            offset_center_y: int = 0
    ) -> "FoliumScreenshotter":
        """Set the screenshot crop frame.

        Args:
            width: Frame width in CSS pixels.
            height: Frame height in CSS pixels.
            offset_center_x: Horizontal offset from screen center.
            offset_center_y: Vertical offset from screen center.

        Returns:
            Self for method chaining.
        """
        self._frame_config = FrameConfig(width, height, offset_center_x, offset_center_y)
        return self

    def set_map_view(self, lat: float, lng: float, zoom: float | None = None) -> "FoliumScreenshotter":
        """Set the map view center and zoom level.

        Args:
            lat: Center latitude.
            lng: Center longitude.
            zoom: Zoom level. Supports fractional values.

        Returns:
            Self for method chaining.
        """
        self._map_view_config = MapViewConfig(lat, lng, zoom)
        return self

    def capture_all_base_layers(self, output_dir: str | Path) -> list[Path]:
        """Capture screenshots of all base layers (overlay=False feature groups).

        Iterates through each base layer in the Leaflet layer control,
        selects it, and takes a screenshot. UI elements (layer control, legends)
        are hidden during capture.

        Args:
            output_dir: Directory to save screenshots. Created if it doesn't exist.

        Returns:
            List of paths to saved screenshot files.

        Raises:
            RuntimeError: If no Leaflet map object is found in the HTML.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        saved_files = []

        try:
            self._start_browser()
            self._load_map()

            map_name = self._get_map_object_name()
            if not map_name:
                raise RuntimeError("Could not find Leaflet map object")

            self._set_view(map_name)
            base_layers = self._get_base_layers()

            if not base_layers:
                print("No base layers (overlay=False) found")
                return saved_files

            for layer in base_layers:
                self._select_base_layer(layer["index"])
                filename = f"{self._sanitize_filename(layer['name'])}.png"
                output_path = output_dir / filename
                self._take_screenshot(output_path)
                saved_files.append(output_path)
                print(f"Saved: {output_path}")

            return saved_files
        finally:
            self._stop_browser()

    def list_exclusive_feature_groups(self) -> list[str]:
        """List all exclusive feature groups (overlay=False).

        Returns:
            List of exclusive feature group names.
        """
        try:
            self._start_browser()
            self._load_map()

            return self._driver.execute_script("""
                let result = [];
                document.querySelectorAll('.leaflet-control-layers-base label').forEach(label => {
                    result.push(label.textContent.trim());
                });
                return result;
            """)
        finally:
            self._stop_browser()

    def list_overlay_feature_groups(self) -> list[str]:
        """List all overlay feature groups (overlay=True).

        Returns:
            List of overlay feature group names.
        """
        try:
            self._start_browser()
            self._load_map()

            return self._driver.execute_script("""
                let result = [];
                document.querySelectorAll('.leaflet-control-layers-overlays label').forEach(label => {
                    result.push(label.textContent.trim());
                });
                return result;
            """)
        finally:
            self._stop_browser()

    def _set_overlay_states(self, active_overlays: list[str]) -> None:
        """Set which overlays are visible.

        Args:
            active_overlays: List of overlay names that should be visible.
                            All others will be hidden.
        """
        self._driver.execute_script("""
            let activeNames = arguments[0];
            document.querySelectorAll('.leaflet-control-layers-overlays label').forEach(label => {
                let input = label.querySelector('input');
                let name = label.textContent.trim();
                let shouldBeChecked = activeNames.includes(name);

                if (input.checked !== shouldBeChecked) {
                    input.click();
                }
            });
        """, active_overlays)
        time.sleep(0.5)

    def capture_selected_exclusives(
            self,
            layer_names: list[str],
            output_dir: str | Path,
            screen_config: ScreenConfig | None = None,
            frame_config: FrameConfig | None = None,
            map_view_config: MapViewConfig | None = None,
            active_overlays: list[str] | None = None,
    ) -> list[Path]:
        """Capture screenshots of specific exclusive layers by name.

        Args:
            layer_names: List of exclusive layer names to capture.
            output_dir: Directory to save screenshots.
            screen_config: Temporary screen config (overrides instance config).
            frame_config: Temporary frame config (overrides instance config).
            map_view_config: Temporary map view config (overrides instance config).
            active_overlays: List of overlay names to keep visible during capture.
                            If None, all overlays remain in their current state.

        Returns:
            List of paths to saved screenshot files.

        Raises:
            ValueError: If a specified layer name is not found.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        saved_files = []

        # Use temporary configs if provided, otherwise fall back to instance configs
        temp_screen = screen_config or self._screen_config
        temp_frame = frame_config or self._frame_config
        temp_map_view = map_view_config or self._map_view_config

        # Temporarily swap configs
        original_screen = self._screen_config
        original_frame = self._frame_config
        original_map_view = self._map_view_config

        try:
            self._screen_config = temp_screen
            self._frame_config = temp_frame
            self._map_view_config = temp_map_view

            self._start_browser()
            self._load_map()

            map_name = self._get_map_object_name()
            if not map_name:
                raise RuntimeError("Could not find Leaflet map object")

            self._set_view(map_name)

            if active_overlays is not None:
                self._set_overlay_states(active_overlays)

            base_layers = self._get_base_layers()
            layer_map = {layer["name"]: layer for layer in base_layers}

            for name in layer_names:
                if name not in layer_map:
                    raise ValueError(f"Layer '{name}' not found. Available: {list(layer_map.keys())}")

                layer = layer_map[name]
                self._select_base_layer(layer["index"])
                filename = f"{self._sanitize_filename(name)}.png"
                output_path = output_dir / filename
                self._take_screenshot(output_path)
                saved_files.append(output_path)
                print(f"Saved: {output_path}")

            return saved_files
        finally:
            # Restore original configs
            self._screen_config = original_screen
            self._frame_config = original_frame
            self._map_view_config = original_map_view
            self._stop_browser()

    def capture_single_view(self, output_path: str | Path) -> Path:
        """Capture a single screenshot of the current map view.

        Takes a screenshot with the currently active base layer.
        UI elements (layer control, legends) are hidden during capture.

        Args:
            output_path: Path for the output PNG file.

        Returns:
            Path to the saved screenshot file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            self._start_browser()
            self._load_map()

            map_name = self._get_map_object_name()
            if map_name:
                self._set_view(map_name)

            self._take_screenshot(output_path)
            return output_path
        finally:
            self._stop_browser()

    def capture_legends(self, output_dir: str | Path) -> dict[str, Path]:
        """Capture screenshots of all detected legend elements.

        Detects legends by ID pattern (`*Legend_*`) or by structure
        (position:fixed elements containing `.legend-content`).
        Each legend is captured individually with other legends hidden.

        Args:
            output_dir: Directory to save legend screenshots.

        Returns:
            Dictionary mapping legend names (title or element ID) to file paths.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        saved_files = {}

        try:
            self._start_browser()
            self._load_map()

            legends = self._detect_legends()

            if not legends:
                print("No legends detected")
                return saved_files

            for legend in legends:
                name = legend.title or legend.element_id
                filename = f"{self._sanitize_filename(name)}.png"
                output_path = output_dir / filename
                self._take_element_screenshot(legend.element_id, output_path)
                saved_files[name] = output_path
                print(f"Saved legend: {output_path} ({legend.width}x{legend.height}px)")

            return saved_files
        finally:
            self._stop_browser()

    def capture_all(self, output_dir: str | Path) -> dict[str, list[Path] | dict[str, Path]]:
        """Capture screenshots of all base layers and legends.

        Convenience method that captures everything in a single browser session.
        Outputs are organized into subdirectories: `layers/` for base layer
        screenshots and `legends/` for legend screenshots.

        Args:
            output_dir: Root directory for output. Subdirectories are created automatically.

        Returns:
            Dictionary with keys:

            - `base_layers`: List of paths to layer screenshots.
            - `legends`: Dictionary mapping legend names to file paths.

        Raises:
            RuntimeError: If no Leaflet map object is found in the HTML.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {
            "base_layers": [],
            "legends": {}
        }

        try:
            self._start_browser()
            self._load_map()

            map_name = self._get_map_object_name()
            if not map_name:
                raise RuntimeError("Could not find Leaflet map object")

            self._set_view(map_name)

            legends = self._detect_legends()
            legends_dir = output_dir / "legends"
            legends_dir.mkdir(exist_ok=True)

            for legend in legends:
                name = legend.title or legend.element_id
                filename = f"{self._sanitize_filename(name)}.png"
                output_path = legends_dir / filename
                self._take_element_screenshot(legend.element_id, output_path)
                results["legends"][name] = output_path
                print(f"Saved legend: {output_path}")

            base_layers = self._get_base_layers()
            layers_dir = output_dir / "layers"
            layers_dir.mkdir(exist_ok=True)

            for layer in base_layers:
                self._select_base_layer(layer["index"])
                filename = f"{self._sanitize_filename(layer['name'])}.png"
                output_path = layers_dir / filename
                self._take_screenshot(output_path)
                results["base_layers"].append(output_path)
                print(f"Saved layer: {output_path}")

            return results
        finally:
            self._stop_browser()


if __name__ == "__main__":
    import folium
    from branca.element import MacroElement, Template


    class DummyLegend(MacroElement):
        _template = Template("""
        {% macro header(this, kwargs) %}
            <style>
                #{{ this.get_name() }} {
                    position: fixed;
                    bottom: 20px;
                    right: 20px;
                    z-index: 1000;
                    background: white;
                    padding: 10px;
                    border-radius: 5px;
                    box-shadow: 0 0 5px rgba(0,0,0,0.2);
                }
                #{{ this.get_name() }} .legend-title {
                    font-weight: bold;
                    margin-bottom: 5px;
                }
            </style>
        {% endmacro %}
        {% macro html(this, kwargs) %}
            <div id="{{ this.get_name() }}">
                <div class="legend-title">{{ this.title }}</div>
                <div class="legend-content">
                    <div style="background: linear-gradient(to right, blue, green, yellow, red); height: 20px; width: 150px;"></div>
                    <div style="display: flex; justify-content: space-between;"><span>0</span><span>100</span></div>
                </div>
            </div>
        {% endmacro %}
        """)

        def __init__(self, title: str):
            super().__init__()
            self._name = f"DummyLegend_{id(self)}"
            self.title = title


    m = folium.Map(location=[52.52, 13.405], zoom_start=10)

    fg1 = folium.FeatureGroup(name="Price Zones", overlay=False)
    folium.CircleMarker([52.52, 13.405], radius=20, color="red").add_to(fg1)
    fg1.add_to(m)

    fg2 = folium.FeatureGroup(name="Flow Lines", overlay=False)
    folium.PolyLine([[52.52, 13.405], [52.45, 13.5]], color="blue", weight=4).add_to(fg2)
    fg2.add_to(m)

    legend = DummyLegend("Net Position [MW]")
    legend.add_to(m)

    folium.LayerControl().add_to(m)

    html_path = Path("test_map.html")
    m.save(str(html_path))
    print(f"Created test map: {html_path}")

    screenshotter = FoliumScreenshotter(
        html_path,
        screen_config=ScreenConfig(1920, 1080, 2.0),
        frame_config=FrameConfig(800, 600),
        map_view_config=MapViewConfig(52, 13, 6),
    )

    legends = screenshotter.capture_legends("screenshots/legends_only")
    print(f"\nCaptured {len(legends)} legends")

    screenshotter2 = FoliumScreenshotter(
        html_path,
        screen_config=ScreenConfig(1920, 1080, 2.0),
        frame_config=FrameConfig(800, 600),
        map_view_config=MapViewConfig(52, 13, 6),
    )
    results = screenshotter2.capture_all("screenshots/all")
    print(f"\nCaptured {len(results['base_layers'])} layers and {len(results['legends'])} legends :)")