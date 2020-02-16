from zilliz_gis.util.vega.scatter_plot.vega_circle_2d import VegaCircle2d
from zilliz_gis.util.vega.heat_map.vega_heat_map import VegaHeatMap

import json
import pytest

def test_vega_circle2d():
    vega_circle2d = VegaCircle2d(1900, 1410, 3, "#2DEF4A", 0.5)
    vega_json = vega_circle2d.build()
    vega_dict = json.loads(vega_json)
    assert vega_dict["width"] == 1900
    assert vega_dict["height"] == 1410
    assert vega_dict["marks"][0]["encode"]["enter"]["shape"]["value"] == "circle"
    assert vega_dict["marks"][0]["encode"]["enter"]["stroke"]["value"] == "#2DEF4A"
    assert vega_dict["marks"][0]["encode"]["enter"]["strokeWidth"]["value"] == 3
    assert vega_dict["marks"][0]["encode"]["enter"]["opacity"]["value"] == 0.5

def test_vega_heat_map():
    vega_heat_map = VegaHeatMap(1900, 1410, 10.0)
    vega_json = vega_heat_map.build()
    vega_dict = json.loads(vega_json)
    assert vega_dict["width"] == 1900
    assert vega_dict["height"] == 1410
    assert vega_dict["marks"][0]["encode"]["enter"]["map_scale"]["value"] == 10.0