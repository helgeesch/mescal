import os

import pandas as pd
import geopandas as gpd
import folium


class MapCountryPlotter:
    GEOJSON_ID_COLUMNS = ['ISO_A2', 'ISO_A3', 'SOV_A3', 'FIPS_10_']
    DEFAULT_STYLE = {
        'fillColor': '#C2C2C2',
        'color': 'white',
        'weight': 0.5,
        'fillOpacity': 1
    }

    def __init__(self, geojson_file_path: str = None):
        if geojson_file_path is None:
            gdf = _load_countries_geojson()
        else:
            gdf = gpd.read_file(geojson_file_path)
        gdf.loc[:, 'geometry'] = gdf.buffer(0)
        gdf.set_crs('EPSG:4326', allow_override=True, inplace=True)
        self._countries_gdf = gdf

    def add_countries_to_feature_group(
            self,
            fg: folium.FeatureGroup,
            countries: list[str],
            style: dict = None
    ) -> folium.FeatureGroup:
        countries_gdf = self._find_countries(countries)
        if countries_gdf.empty:
            return fg

        style_dict = self._get_updated_style_dict(style)
        folium.GeoJson(countries_gdf, style_function=lambda x: style_dict).add_to(fg)
        return fg

    def _get_updated_style_dict(self, style: dict = None) -> dict:
        style_dict = self.DEFAULT_STYLE.copy()
        if style:
            style_dict.update(style)
        return style_dict

    def add_all_countries_except(
            self,
            fg: folium.FeatureGroup,
            excluded_countries: list[str],
            style: dict = None
    ) -> folium.FeatureGroup:
        excluded_gdf = self._find_countries(excluded_countries)
        all_countries = self._countries_gdf.copy()

        if not excluded_gdf.empty:
            all_countries = all_countries[~all_countries.index.isin(excluded_gdf.index)]

        style_dict = self._get_updated_style_dict(style)
        folium.GeoJson(all_countries, style_function=lambda x: style_dict).add_to(fg)
        return fg

    def get_geojson_for_country(self, country_id: str) -> gpd.GeoDataFrame:
        for column in self.GEOJSON_ID_COLUMNS:
            if column not in self._countries_gdf.columns:
                continue
            match = self._countries_gdf[self._countries_gdf[column] == country_id]
            if not match.empty:
                return match
        return gpd.GeoDataFrame()

    def _find_countries(self, countries: list[str]) -> gpd.GeoDataFrame:
        matches = gpd.GeoDataFrame()
        for country in countries:
            country_match = self.get_geojson_for_country(country)
            if not country_match.empty:
                matches = pd.concat([matches, country_match])
        return matches


def _load_countries_geojson() -> gpd.GeoDataFrame:
    from pathlib import Path
    from importlib.resources import files
    from mesqual.utils.package_path import get_abs_source_root_path

    # Try local dev path (source root)
    local_path = Path(os.path.join(get_abs_source_root_path(), "mesqual/data/countries.geojson"))
    if local_path.exists():
        return gpd.read_file(local_path)

    # Fallback to installed package path
    try:
        pkg_path = files("mesqual.data").joinpath("countries.geojson")
        return gpd.read_file(str(pkg_path))
    except Exception as e:
        raise FileNotFoundError("❌ Could not locate 'countries.geojson' in either local or package paths.") from e