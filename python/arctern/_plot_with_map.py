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
import io
import base64
from arctern.util.vega import vega_pointmap, vega_weighted_pointmap, vega_heatmap, vega_choroplethmap, vega_icon, vega_fishnetmap
import arctern


def _get_recom_size(dx, dy, target=(1600, 1600)):
    scale_x = target[0] / dx
    scale_y = target[1] / dy
    scale = min(scale_x, scale_y)
    w = int(dx * scale)
    h = int(dy * scale)
    return w, h


def _transform_bbox(bounding_box, src_coord_sys, dst_coord_sys):
    import pyproj
    if src_coord_sys != dst_coord_sys:
        x0, y0, x1, y1 = bounding_box
        dst_proj = pyproj.Proj(dst_coord_sys)
        src_proj = pyproj.Proj(src_coord_sys)
        x0, y0 = pyproj.transform(src_proj, dst_proj, x0, y0, always_xy=True)
        x1, y1 = pyproj.transform(src_proj, dst_proj, x1, y1, always_xy=True)
        bounding_box = (x0, y0, x1, y1)
    return bounding_box


def plot_pointmap(ax, points, bounding_box,
                  point_size=3, point_color='#115f9a', opacity=1.0,
                  coordinate_system='EPSG:3857',
                  **extra_contextily_params):
    """
    :type ax: AxesSubplot
    :param ax: Matplotlib axes object on which to add the basemap.

    :type points: Series(dtype: object)
    :param points: Points in WKB form
    :type bounding_box: (float, float, float, float)
    :param bounding_box: The bounding rectangle, as a [left, upper, right, lower]-tuple.
                        value should be of :coordinate_system:
    :type point_szie: int
    :param point_size: size of point
    :type point_color: str
    :param point_color: specify color in hex form
    :type opacity: float
    :param opacity: opacity of point
    :type coordinate_system: str
    :param coordinate_system: either 'EPSG:4326' or 'EPSG:3857'
    :type extra_contextily_params: dict
    :param extra_contextily_params: extra parameters for contextily.add_basemap.
                                                                    See https://contextily.readthedocs.io/en/latest/reference.html
    """
    from matplotlib import pyplot as plt
    import contextily as cx
    bbox = _transform_bbox(bounding_box, coordinate_system, 'epsg:3857')
    w, h = _get_recom_size(bbox[2]-bbox[0], bbox[3]-bbox[1])
    vega = vega_pointmap(w, h, bounding_box=bounding_box, point_size=point_size,
                         point_color=point_color, opacity=opacity, coordinate_system=coordinate_system)
    hexstr = arctern.point_map_layer(vega, points)
    f = io.BytesIO(base64.b64decode(hexstr))

    img = plt.imread(f)
    ax.set(xlim=(bbox[0], bbox[2]), ylim=(bbox[1], bbox[3]))
    cx.add_basemap(ax, **extra_contextily_params)
    ax.imshow(img, alpha=img[:, :, 3], extent=(bbox[0], bbox[2], bbox[1], bbox[3]))

# pylint: disable=too-many-arguments
# pylint: disable=dangerous-default-value


def plot_weighted_pointmap(ax, points, color_weights=None,
                           size_weights=None,
                           bounding_box=None,
                           color_gradient=["#115f9a", "#d0f400"],
                           color_bound=[0, 0],
                           size_bound=[3],
                           opacity=1.0,
                           coordinate_system='EPSG:3857',
                           **extra_contextily_params):
    """
    :type ax: AxesSubplot
    :param ax: Matplotlib axes object on which to add the basemap.

    :type points: Series(dtype: object)
    :param points: Points in WKB form
    :type bounding_box: (float, float, float, float)
    :param bounding_box: The bounding rectangle, as a [left, upper, right, lower]-tuple.
                         value should be of :coordinate_system:
    :type point_szie: int
    :param point_size: size of point
    :type opacity: float
    :param opacity: opacity of point
    :type coordinate_system: str
    :param coordinate_system: either 'EPSG:4326' or 'EPSG:3857'
    :type extra_contextily_params: dict
    :param extra_contextily_params: extra parameters for contextily.add_basemap.
                                    See https://contextily.readthedocs.io/en/latest/reference.html
    """
    from matplotlib import pyplot as plt
    import contextily as cx
    bbox = _transform_bbox(bounding_box, coordinate_system, 'epsg:3857')
    w, h = _get_recom_size(bbox[2]-bbox[0], bbox[3]-bbox[1])
    vega = vega_weighted_pointmap(w, h, bounding_box=bounding_box, color_gradient=color_gradient,
                                  color_bound=color_bound, size_bound=size_bound, opacity=opacity,
                                  coordinate_system=coordinate_system)
    hexstr = arctern.weighted_point_map_layer(
        vega, points, color_weights=color_weights, size_weights=size_weights)
    f = io.BytesIO(base64.b64decode(hexstr))

    img = plt.imread(f)
    ax.set(xlim=(bbox[0], bbox[2]), ylim=(bbox[1], bbox[3]))
    cx.add_basemap(ax, **extra_contextily_params)
    ax.imshow(img, alpha=img[:, :, 3], extent=(bbox[0], bbox[2], bbox[1], bbox[3]))

# pylint: disable=protected-access


def _calc_zoom(bbox, coordinate_system):
    import contextily as cx
    bbox = _transform_bbox(bbox, coordinate_system, 'epsg:4326')
    return cx.tile._calculate_zoom(*bbox)


def plot_heatmap(ax, points, weights, bounding_box,
                 map_zoom_level=None,
                 coordinate_system='EPSG:3857',
                 aggregation_type='max',
                 **extra_contextily_params):
    """
    :type ax: AxesSubplot
    :param ax: Matplotlib axes object on which to add the basemap.

    :type points: Series(dtype: object)
    :param points: Points in WKB form
    :type bounding_box: (float, float, float, float)
    :param bounding_box: The bounding rectangle, as a [left, upper, right, lower]-tuple.
                                             value should be of :coordinate_system:
    :type coordinate_system: str
    :param coordinate_system: either 'EPSG:4326' or 'EPSG:3857'
    :type extra_contextily_params: dict
    :param extra_contextily_params: extra parameters for contextily.add_basemap.
                                                                    See https://contextily.readthedocs.io/en/latest/reference.html
    """
    from matplotlib import pyplot as plt
    import contextily as cx
    bbox = _transform_bbox(bounding_box, coordinate_system, 'epsg:3857')
    w, h = _get_recom_size(bbox[2]-bbox[0], bbox[3]-bbox[1])
    if map_zoom_level is None:
        map_zoom_level = _calc_zoom(bounding_box, coordinate_system)
    vega = vega_heatmap(w, h, bounding_box=bounding_box, map_zoom_level=map_zoom_level,
                        aggregation_type=aggregation_type, coordinate_system=coordinate_system)
    hexstr = arctern.heat_map_layer(vega, points, weights)
    f = io.BytesIO(base64.b64decode(hexstr))

    img = plt.imread(f)
    ax.set(xlim=(bbox[0], bbox[2]), ylim=(bbox[1], bbox[3]))
    cx.add_basemap(ax, **extra_contextily_params)
    ax.imshow(img, alpha=img[:, :, 3], extent=(bbox[0], bbox[2], bbox[1], bbox[3]))


def plot_choroplethmap(ax, region_boundaries, weights, bounding_box,
                       color_gradient, color_bound=None, opacity=1.0,
                       coordinate_system='EPSG:3857',
                       aggregation_type='max',
                       **extra_contextily_params):
    """
    :type ax: AxesSubplot
    :param ax: Matplotlib axes object on which to add the basemap.

    :type points: Series(dtype: object)
    :param points: Points in WKB form
    :type bounding_box: (float, float, float, float)
    :param bounding_box: The bounding rectangle, as a [left, upper, right, lower]-tuple.
                        value should be of :coordinate_system:
    :type coordinate_system: str
    :param coordinate_system: either 'EPSG:4326' or 'EPSG:3857'
    :type extra_contextily_params: dict
    :param extra_contextily_params: extra parameters for contextily.add_basemap.
                                                                    See https://contextily.readthedocs.io/en/latest/reference.html
    """
    from matplotlib import pyplot as plt
    import contextily as cx
    bbox = _transform_bbox(bounding_box, coordinate_system, 'epsg:3857')
    w, h = _get_recom_size(bbox[2]-bbox[0], bbox[3]-bbox[1])
    vega = vega_choroplethmap(w, h, bounding_box=bounding_box, color_gradient=color_gradient, color_bound=color_bound,
                              opacity=opacity, aggregation_type=aggregation_type, coordinate_system=coordinate_system)
    hexstr = arctern.choropleth_map_layer(vega, region_boundaries, weights)
    f = io.BytesIO(base64.b64decode(hexstr))

    img = plt.imread(f)
    ax.set(xlim=(bbox[0], bbox[2]), ylim=(bbox[1], bbox[3]))
    cx.add_basemap(ax, **extra_contextily_params)
    ax.imshow(img, alpha=img[:, :, 3], extent=(bbox[0], bbox[2], bbox[1], bbox[3]))


def plot_iconviz(ax, points, bounding_box, icon_path,
                 coordinate_system='EPSG:3857',
                 **extra_contextily_params):
    """
    :type ax: AxesSubplot
    :param ax: Matplotlib axes object on which to add the basemap.
    :type points: Series(dtype: object)
    :param points: Points in WKB form
    :type bounding_box: (float, float, float, float)
    :param bounding_box: The bounding rectangle, as a [left, upper, right, lower]-tuple.
                                             value should be of :coordinate_system:
    :type coordinate_system: str
    :param coordinate_system: either 'EPSG:4326' or 'EPSG:3857'
    :type extra_contextily_params: dict
    :param extra_contextily_params: extra parameters for contextily.add_basemap.
                                                                    See https://contextily.readthedocs.io/en/latest/reference.html
    """
    from matplotlib import pyplot as plt
    import contextily as cx
    bbox = _transform_bbox(bounding_box, coordinate_system, 'epsg:3857')
    w, h = _get_recom_size(bbox[2]-bbox[0], bbox[3]-bbox[1])
    vega = vega_icon(w, h, bounding_box=bounding_box, icon_path=icon_path,
                     coordinate_system=coordinate_system)
    hexstr = arctern.icon_viz_layer(vega, points)
    f = io.BytesIO(base64.b64decode(hexstr))

    img = plt.imread(f)
    ax.set(xlim=(bbox[0], bbox[2]), ylim=(bbox[1], bbox[3]))
    cx.add_basemap(ax, **extra_contextily_params)
    ax.imshow(img, alpha=img[:, :, 3], extent=(bbox[0], bbox[2], bbox[1], bbox[3]))

# pylint: disable=too-many-arguments
# pylint: disable=dangerous-default-value


def plot_fishnetmap(ax, points, weights, bounding_box,
                    color_gradient=["#0000FF", "#FF0000"],
                    cell_size=4, cell_spacing=1, opacity=1.0,
                    coordinate_system='epsg:3857',
                    aggregation_type='sum',
                    **extra_contextily_params):
    from matplotlib import pyplot as plt
    import contextily as cx
    bbox = _transform_bbox(bounding_box, coordinate_system, 'epsg:3857')
    w, h = _get_recom_size(bbox[2]-bbox[0], bbox[3]-bbox[1])
    vega = vega_fishnetmap(w, h, bounding_box=bounding_box, color_gradient=color_gradient,
                           cell_size=cell_size, cell_spacing=cell_spacing, opacity=opacity,
                           coordinate_system=coordinate_system, aggregation_type=aggregation_type)
    hexstr = arctern.fishnet_map_layer(vega, points, weights)
    f = io.BytesIO(base64.b64decode(hexstr))

    img = plt.imread(f)
    ax.set(xlim=(bbox[0], bbox[2]), ylim=(bbox[1], bbox[3]))
    cx.add_basemap(ax, **extra_contextily_params)
    ax.imshow(img, alpha=img[:, :, 3], extent=(bbox[0], bbox[2], bbox[1], bbox[3]))
