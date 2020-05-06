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

__all__ = [
    "vega_pointmap",
    "vega_weighted_pointmap",
    "vega_heatmap",
    "vega_choroplethmap",
    "vega_icon",
    "vega_fishnetmap"
]

from arctern.util.vega.pointmap.vega_pointmap import VegaPointMap
from arctern.util.vega.pointmap.vega_weighted_pointmap import VegaWeightedPointMap
from arctern.util.vega.heatmap.vega_heatmap import VegaHeatMap
from arctern.util.vega.choroplethmap.vega_choroplethmap import VegaChoroplethMap
from arctern.util.vega.icon.vega_icon import VegaIcon
from arctern.util.vega.fishnetmap.vega_fishnetmap import VegaFishNetMap


def vega_pointmap(width,
                  height,
                  bounding_box,
                  point_size=3,
                  point_color="#115f9a",
                  opacity=1.0,
                  coordinate_system="EPSG:3857"):
    return VegaPointMap(width,
                        height,
                        bounding_box,
                        point_size,
                        point_color,
                        opacity,
                        coordinate_system)


def vega_weighted_pointmap(width,
                           height,
                           bounding_box,
                           color_gradient,
                           color_bound=None,
                           size_bound=None,
                           opacity=1.0,
                           coordinate_system="EPSG:3857",
                           aggregation_type="max"):
    if color_bound is None:
        color_bound = [0, 0]
    if size_bound is None:
        size_bound = [3]
    return VegaWeightedPointMap(width,
                                height,
                                bounding_box,
                                color_gradient,
                                color_bound,
                                size_bound,
                                opacity,
                                coordinate_system,
                                aggregation_type)


def vega_heatmap(width,
                 height,
                 bounding_box,
                 map_zoom_level,
                 coordinate_system="EPSG:3857",
                 aggregation_type="sum"):
    return VegaHeatMap(width,
                       height,
                       bounding_box,
                       map_zoom_level,
                       coordinate_system,
                       aggregation_type)


def vega_choroplethmap(width,
                       height,
                       bounding_box,
                       color_gradient,
                       color_bound=None,
                       opacity=1.0,
                       coordinate_system="EPSG:3857",
                       aggregation_type="sum"):
    if color_bound is None:
        color_bound = [0, 0]
    return VegaChoroplethMap(width,
                             height,
                             bounding_box,
                             color_gradient,
                             color_bound,
                             opacity,
                             coordinate_system,
                             aggregation_type)


def vega_icon(width, height,
              bounding_box, icon_path,
              coordinate_system="EPSG:3857"):
    return VegaIcon(width, height,
                    bounding_box,
                    icon_path,
                    coordinate_system)

def vega_fishnetmap(width,
                   height,
                   bounding_box,
                   color_gradient=None,
                   cell_size=4,
                   cell_spacing=1,
                   opacity=1.0,
                   coordinate_system="EPSG:3857",
                   aggregation_type="sum"):
    if color_gradient is None:
        color_gradient = ["#00FF00", "FF0000"]
    return VegaFishNetMap(width,
                         height,
                         bounding_box,
                         color_gradient,
                         cell_size,
                         cell_spacing,
                         opacity,
                         coordinate_system,
                         aggregation_type)
