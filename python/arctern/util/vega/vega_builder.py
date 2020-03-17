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

from arctern.util.vega.scatter_plot.vega_circle_2d import VegaCircle2d
from arctern.util.vega.heat_map.vega_heat_map import VegaHeatMap
from arctern.util.vega.choropleth_map.choropleth_map import VegaChoroplethMap

def vega_circle2d(width, height,
                  bounding_box_min, bounding_box_max,
                  stroke, stroke_width, opacity,
                  coordinate_system="EPSG:4326"):
    vega = VegaCircle2d(width, height,
                        bounding_box_min, bounding_box_max,
                        stroke, stroke_width, opacity,
                        coordinate_system)
    return vega.build().encode('utf-8')

def vega_heatmap(width, height,
                 bounding_box_min, bounding_box_max,
                 map_scale,
                 coordinate_system="EPSG:4326"):
    vega = VegaHeatMap(width, height,
                       bounding_box_min, bounding_box_max,
                       map_scale,
                       coordinate_system)
    return vega.build().encode('utf-8')

def vega_choroplethmap(width, height,
                       bounding_box_min, bounding_box_max,
                       color_style, ruler, opacity,
                       coordinate_system="EPSG:4326"):
    vega = VegaChoroplethMap(width, height,
                             bounding_box_min, bounding_box_max, color_style, ruler, opacity,
                             coordinate_system)
    return vega.build().encode('utf-8')
