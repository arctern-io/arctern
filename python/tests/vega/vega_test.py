# Copyright (C) 2019-2020 Zilliz. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from zilliz_gis.util.vega.scatter_plot.vega_circle_2d import VegaCircle2d
from zilliz_gis.util.vega.heat_map.vega_heat_map import VegaHeatMap
from zilliz_gis.util.vega.choropleth_map.choropleth_map import VegaChoroplethMap


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

def test_vega_choropleth_map():
    vega_choropleth_map = VegaChoroplethMap(1900, 1410,
                                            [-73.984092, 40.753893, -73.977588, 40.756342],
                                            "blue_to_red", [2.5, 5], 1.0)
    vega_json = vega_choropleth_map.build()
    vega_dict = json.loads(vega_json)
    assert vega_dict["width"] == 1900
    assert vega_dict["height"] == 1410
    assert len(vega_dict["marks"][0]["encode"]["enter"]["bounding_box"]["value"]) == 4
    assert vega_dict["marks"][0]["encode"]["enter"]["bounding_box"]["value"][0] == -73.984092
    assert vega_dict["marks"][0]["encode"]["enter"]["bounding_box"]["value"][1] == 40.753893
    assert vega_dict["marks"][0]["encode"]["enter"]["bounding_box"]["value"][2] == -73.977588
    assert vega_dict["marks"][0]["encode"]["enter"]["bounding_box"]["value"][3] == 40.756342
    assert vega_dict["marks"][0]["encode"]["enter"]["color_style"]["value"] == "blue_to_red"
    assert len(vega_dict["marks"][0]["encode"]["enter"]["ruler"]["value"]) == 2
    assert vega_dict["marks"][0]["encode"]["enter"]["ruler"]["value"][0] == 2.5
    assert vega_dict["marks"][0]["encode"]["enter"]["ruler"]["value"][1] == 5
    assert vega_dict["marks"][0]["encode"]["enter"]["opacity"]["value"] == 1.0
