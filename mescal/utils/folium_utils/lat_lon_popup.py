import folium


class CustomLatLngPopup(folium.LatLngPopup):
    """Module enables a simple Popup showing the lat and lon coordinates upon a click anywhere in the map."""
    _template = folium.features.Template("""
        {% macro script(this, kwargs) %}
            var {{this.get_name()}} = L.popup();
            function latLngPop(e) {
                {{this.get_name()}}
                    .setLatLng(e.latlng)
                    .setContent("Latitude: " + e.latlng.lat.toFixed(4) +
                                "<br>Longitude: " + e.latlng.lng.toFixed(4) +
                                "<br><br>POINT("+e.latlng.lng.toFixed(4)+" "+e.latlng.lat.toFixed(4)+")")
                    .openOn({{this._parent.get_name()}});
                }
            {{this._parent.get_name()}}.on('click', latLngPop);
        {% endmacro %}
    """)


if __name__ == '__main__':
    m = folium.Map()
    m.add_child(CustomLatLngPopup())
    m.save(f'_tmp/map_lat_lon_popup.html')
