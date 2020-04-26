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
from arctern.util.vega import vega_pointmap, vega_weighted_pointmap, vega_heatmap, vega_choroplethmap, vega_icon


def test_vega_pointmap():
    vega = vega_pointmap(1900, 1410, [-73.998427, 40.730309, -73.954348, 40.780816], 3, "#2DEF4A", 0.5, "EPSG:3857").build()
    vega_dict = json.loads(vega)
    assert vega_dict["width"] == 1900
    assert vega_dict["height"] == 1410
    assert vega_dict["marks"][0]["encode"]["enter"]["bounding_box"]["value"][0] == -73.998427
    assert vega_dict["marks"][0]["encode"]["enter"]["bounding_box"]["value"][1] == 40.730309
    assert vega_dict["marks"][0]["encode"]["enter"]["bounding_box"]["value"][2] == -73.954348
    assert vega_dict["marks"][0]["encode"]["enter"]["bounding_box"]["value"][3] == 40.780816
    assert vega_dict["marks"][0]["encode"]["enter"]["shape"]["value"] == "circle"
    assert vega_dict["marks"][0]["encode"]["enter"]["point_color"]["value"] == "#2DEF4A"
    assert vega_dict["marks"][0]["encode"]["enter"]["point_size"]["value"] == 3
    assert vega_dict["marks"][0]["encode"]["enter"]["opacity"]["value"] == 0.5
    assert vega_dict["marks"][0]["encode"]["enter"]["coordinate_system"]["value"] == "EPSG:3857"

    vega = vega_pointmap(1900, 1410, [-73.998427, 40.730309, -73.954348, 40.780816]).build()
    vega_dict = json.loads(vega)
    assert vega_dict["marks"][0]["encode"]["enter"]["point_color"]["value"] == "#115f9a"
    assert vega_dict["marks"][0]["encode"]["enter"]["point_size"]["value"] == 3
    assert vega_dict["marks"][0]["encode"]["enter"]["opacity"]["value"] == 1.0
    assert vega_dict["marks"][0]["encode"]["enter"]["coordinate_system"]["value"] == "EPSG:3857"


def test_vega_weighted_pointmap():
    vega = vega_weighted_pointmap(1900, 1410, [-73.998427, 40.730309, -73.954348, 40.780816], ["#2DEF4A"], [2, 5], [1, 10], 0.5, "EPSG:3857", "sum").build()
    vega_dict = json.loads(vega)
    assert vega_dict["width"] == 1900
    assert vega_dict["height"] == 1410
    assert vega_dict["marks"][0]["encode"]["enter"]["bounding_box"]["value"][0] == -73.998427
    assert vega_dict["marks"][0]["encode"]["enter"]["bounding_box"]["value"][1] == 40.730309
    assert vega_dict["marks"][0]["encode"]["enter"]["bounding_box"]["value"][2] == -73.954348
    assert vega_dict["marks"][0]["encode"]["enter"]["bounding_box"]["value"][3] == 40.780816
    assert vega_dict["marks"][0]["encode"]["enter"]["color_gradient"]["value"][0] == "#2DEF4A"
    assert vega_dict["marks"][0]["encode"]["enter"]["color_bound"]["value"][0] == 2
    assert vega_dict["marks"][0]["encode"]["enter"]["color_bound"]["value"][1] == 5
    assert vega_dict["marks"][0]["encode"]["enter"]["size_bound"]["value"][0] == 1
    assert vega_dict["marks"][0]["encode"]["enter"]["size_bound"]["value"][1] == 10
    assert vega_dict["marks"][0]["encode"]["enter"]["opacity"]["value"] == 0.5
    assert vega_dict["marks"][0]["encode"]["enter"]["coordinate_system"]["value"] == "EPSG:3857"
    assert vega_dict["marks"][0]["encode"]["enter"]["aggregation_type"]["value"] == "sum"

    vega = vega_weighted_pointmap(1900, 1410, [-73.998427, 40.730309, -73.954348, 40.780816], ["#2DEF4A"]).build()
    vega_dict = json.loads(vega)
    assert vega_dict["marks"][0]["encode"]["enter"]["color_bound"]["value"][0] == 0
    assert vega_dict["marks"][0]["encode"]["enter"]["color_bound"]["value"][1] == 0
    assert vega_dict["marks"][0]["encode"]["enter"]["size_bound"]["value"][0] == 3
    assert vega_dict["marks"][0]["encode"]["enter"]["opacity"]["value"] == 1.0
    assert vega_dict["marks"][0]["encode"]["enter"]["coordinate_system"]["value"] == "EPSG:3857"
    assert vega_dict["marks"][0]["encode"]["enter"]["aggregation_type"]["value"] == "max"


def test_vega_heat_map():
    vega = vega_heatmap(1900, 1410, [-73.998427, 40.730309, -73.954348, 40.780816], 10.0, "EPSG:3857", "max").build()
    vega_dict = json.loads(vega)
    assert vega_dict["width"] == 1900
    assert vega_dict["height"] == 1410
    assert vega_dict["marks"][0]["encode"]["enter"]["bounding_box"]["value"][0] == -73.998427
    assert vega_dict["marks"][0]["encode"]["enter"]["bounding_box"]["value"][1] == 40.730309
    assert vega_dict["marks"][0]["encode"]["enter"]["bounding_box"]["value"][2] == -73.954348
    assert vega_dict["marks"][0]["encode"]["enter"]["bounding_box"]["value"][3] == 40.780816
    assert vega_dict["marks"][0]["encode"]["enter"]["map_zoom_level"]["value"] == 10.0
    assert vega_dict["marks"][0]["encode"]["enter"]["coordinate_system"]["value"] == "EPSG:3857"
    assert vega_dict["marks"][0]["encode"]["enter"]["aggregation_type"]["value"] == "max"

    vega = vega_heatmap(1900, 1410, [-73.998427, 40.730309, -73.954348, 40.780816], 10.0).build()
    vega_dict = json.loads(vega)
    assert vega_dict["marks"][0]["encode"]["enter"]["coordinate_system"]["value"] == "EPSG:3857"
    assert vega_dict["marks"][0]["encode"]["enter"]["aggregation_type"]["value"] == "sum"


def test_vega_choropleth_map():
    vega = vega_choroplethmap(1900, 1410, [-73.994092, 40.753893, -73.977588, 40.759642], ["#0000FF", "#FF0000"], [2.5, 5], 1.0, "EPSG:3857", "max").build()
    vega_dict = json.loads(vega)
    assert vega_dict["width"] == 1900
    assert vega_dict["height"] == 1410
    assert vega_dict["marks"][0]["encode"]["enter"]["bounding_box"]["value"][0] == -73.994092
    assert vega_dict["marks"][0]["encode"]["enter"]["bounding_box"]["value"][1] == 40.753893
    assert vega_dict["marks"][0]["encode"]["enter"]["bounding_box"]["value"][2] == -73.977588
    assert vega_dict["marks"][0]["encode"]["enter"]["bounding_box"]["value"][3] == 40.759642
    assert vega_dict["marks"][0]["encode"]["enter"]["color_gradient"]["value"][0] == "#0000FF"
    assert vega_dict["marks"][0]["encode"]["enter"]["color_gradient"]["value"][1] == "#FF0000"
    assert len(vega_dict["marks"][0]["encode"]["enter"]["color_bound"]["value"]) == 2
    assert vega_dict["marks"][0]["encode"]["enter"]["color_bound"]["value"][0] == 2.5
    assert vega_dict["marks"][0]["encode"]["enter"]["color_bound"]["value"][1] == 5
    assert vega_dict["marks"][0]["encode"]["enter"]["opacity"]["value"] == 1.0
    assert vega_dict["marks"][0]["encode"]["enter"]["coordinate_system"]["value"] == "EPSG:3857"
    assert vega_dict["marks"][0]["encode"]["enter"]["aggregation_type"]["value"] == "max"

    vega = vega_choroplethmap(1900, 1410, [-73.994092, 40.753893, -73.977588, 40.759642], ["#0000FF", "#FF0000"]).build()
    vega_dict = json.loads(vega)
    assert len(vega_dict["marks"][0]["encode"]["enter"]["color_bound"]["value"]) == 2
    assert vega_dict["marks"][0]["encode"]["enter"]["color_bound"]["value"][0] == 0
    assert vega_dict["marks"][0]["encode"]["enter"]["color_bound"]["value"][1] == 0
    assert vega_dict["marks"][0]["encode"]["enter"]["opacity"]["value"] == 1.0
    assert vega_dict["marks"][0]["encode"]["enter"]["coordinate_system"]["value"] == "EPSG:3857"
    assert vega_dict["marks"][0]["encode"]["enter"]["aggregation_type"]["value"] == "sum"


def test_vega_icon():
    vega = vega_icon(1900, 1410, [-73.998427, 40.730309, -73.954348, 40.780816], "icon_path", "EPSG:3857").build()
    vega_dict = json.loads(vega)
    assert vega_dict["width"] == 1900
    assert vega_dict["height"] == 1410
    assert vega_dict["marks"][0]["encode"]["enter"]["bounding_box"]["value"][0] == -73.998427
    assert vega_dict["marks"][0]["encode"]["enter"]["bounding_box"]["value"][1] == 40.730309
    assert vega_dict["marks"][0]["encode"]["enter"]["bounding_box"]["value"][2] == -73.954348
    assert vega_dict["marks"][0]["encode"]["enter"]["bounding_box"]["value"][3] == 40.780816
    assert vega_dict["marks"][0]["encode"]["enter"]["icon_path"]["value"] == "icon_path"
    assert vega_dict["marks"][0]["encode"]["enter"]["coordinate_system"]["value"] == "EPSG:3857"

    vega = vega_icon(1900, 1410, [-73.998427, 40.730309, -73.954348, 40.780816], "icon_path").build()
    vega_dict = json.loads(vega)
    assert vega_dict["marks"][0]["encode"]["enter"]["coordinate_system"]["value"] == "EPSG:3857"
