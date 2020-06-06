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


def pointmap(ax, points, bounding_box,
                  point_size=3, point_color='#115f9a', opacity=1.0,
                  coordinate_system='EPSG:3857',
                  **extra_contextily_params):
    """
    Plot pointmap in Matplotlib
    :type ax: AxesSubplot
    :param ax: Matplotlib axes object on which to add the basemap.

    :type points: GeoSeries
    :param points: Sequence of Points

    :type bounding_box: list
    :param bounding_box: Specify the bounding rectangle [west, south, east, north].

    :type point_size: int
    :param point_size: Diameter of point, default as 3 

    :type point_color: str
    :param point_color: Specify point color in Hex Color Code, default as "#115f9a"

    :type opacity: float
    :param opacity: Opacity of point, ranged from 0.0 to 1.0, default as 1.0

    :type coordinate_system: str
    :param coordinate_system: Coordinate Reference System of the geometry objects.
                    Must be SRID formed, e.g. 'EPSG:4326' or 'EPSG:3857'

    :type extra_contextily_params: dict
    :param extra_contextily_params: Extra parameters will be passed to contextily.add_basemap.
                                    See https://contextily.readthedocs.io/en/latest/reference.html for details

    :example:
    >>> import pandas as pd
    >>> import numpy as np
    >>> import arctern
    >>> import matplotlib.pyplot as plt
    >>> # read from test.csv
    >>> # Download link: https://raw.githubusercontent.com/arctern-io/arctern-resources/benchmarks/benchmarks/dataset/layer_rendering_test_data/test_data.csv
    >>> df = pd.read_csv("/path/to/test_data.csv", dtype={'longitude':np.float64, 'latitude':np.float64, 'color_weights':np.float64, 'size_weights':np.float64, 'region_boundaries':np.object})
    >>> points = arctern.GeoSeries.point(df['longitude'], df['latitude'])
    >>> # plot pointmap
    >>> fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
    >>> arctern.plot.pointmap(ax, points, [-74.01398981737215,40.71353244267465,-73.96979949831308,40.74480271529791], point_size=10, point_color='#115f9a',coordinate_system="EPSG:4326")
    >>> plt.show()
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
    ax.axis('off')

# pylint: disable=too-many-arguments
# pylint: disable=dangerous-default-value


def weighted_pointmap(ax, points, color_weights=None,
                           size_weights=None,
                           bounding_box=None,
                           color_gradient=["#115f9a", "#d0f400"],
                           color_bound=[0, 0],
                           size_bound=[3],
                           opacity=1.0,
                           coordinate_system='EPSG:3857',
                           **extra_contextily_params):
    """
    Plot weighted pointmap in Matplotlib
    :type ax: AxesSubplot
    :param ax: Matplotlib axes object on which to add the basemap.

    :type points: GeoSeries
    :param points: Sequence of Points

    :type color_weights: Series(dtype: float|int64)
    :param color_weights: Weights for point color

    :type size_weights: Series(dtype: float|int64)
    :param size_weights: Weights for point size, deciding diameter of point (after bounded by size_bound)

    :type bounding_box: list
    :param bounding_box: Specify the bounding rectangle [west, south, east, north].

    :type color_gradient: list
    :param color_gradient: Specify range of color gradient. 
                           Either use ["hex_color"] to specify a same color for all points, 
                           or ["hex_color1", "hex_color2"] to specify a color gradient ranging from "hex_color1" to "hex_color2"
    
    :type color_bound: list
    :param color_bound: Specify weight range [w1, w2] binding to color_gradient. 
                        Needed only when color_gradient has two value ["color1", "color2"]. 
                        Bind w1 to "color1", and w2 to "color2".
                        When weight < w1 or weight > w2, truncate to w1/w2 accordingly. 

    :type size_bound: list
    :param size_bound: Specify range [w1, w2] of size_weights. 
                       When weight < w1 or weight > w2, truncate to w1/w2 accordingly. 

    :type opacity: float
    :param opacity: Opacity of point, ranged from 0.0 to 1.0, default as 1.0

    :type coordinate_system: str
    :param coordinate_system: Coordinate Reference System of the geometry objects.
                    Must be SRID formed, e.g. 'EPSG:4326' or 'EPSG:3857'

    :type extra_contextily_params: dict
    :param extra_contextily_params: Extra parameters will be passed to contextily.add_basemap.
                                    See https://contextily.readthedocs.io/en/latest/reference.html for details

    :example:
    >>> import pandas as pd
    >>> import numpy as np
    >>> import arctern
    >>> import matplotlib.pyplot as plt
    >>> # read from test.csv
    >>> # Download link: https://raw.githubusercontent.com/arctern-io/arctern-resources/benchmarks/benchmarks/dataset/layer_rendering_test_data/test_data.csv
    >>> df = pd.read_csv("/path/to/test_data.csv", dtype={'longitude':np.float64, 'latitude':np.float64, 'color_weights':np.float64, 'size_weights':np.float64, 'region_boundaries':np.object})
    >>> points = arctern.GeoSeries.point(df['longitude'], df['latitude'])
    >>> 
    >>> # plot weighted pointmap with variable color and fixed size
    >>> fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
    >>> arctern.plot.weighted_pointmap(ax, points, color_weights=df['color_weights'], bounding_box=[-73.99668712186558,40.72972339069935,-73.99045479584949,40.7345193345495], color_gradient=["#115f9a", "#d0f400"], color_bound=[2.5,15], size_bound=[16], opacity=1.0, coordinate_system="EPSG:4326")
    >>> plt.show()
    >>> 
    >>> # plot weighted pointmap with fixed color and variable size
    >>> fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
    >>> arctern.plot.weighted_pointmap(ax, points, size_weights=df['size_weights'], bounding_box=[-73.99668712186558,40.72972339069935,-73.99045479584949,40.7345193345495], color_gradient=["#37A2DA"], size_bound=[15, 50], opacity=1.0, coordinate_system="EPSG:4326")
    >>> plt.show()
    >>> 
    >>> # plot weighted pointmap with variable color and size
    >>> fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
    >>> arctern.plot.weighted_pointmap(ax, points, color_weights=df['color_weights'], size_weights=df['size_weights'], bounding_box=[-73.99668712186558,40.72972339069935,-73.99045479584949,40.7345193345495], color_gradient=["#115f9a", "#d0f400"], color_bound=[2.5,15], size_bound=[15, 50], opacity=1.0, coordinate_system="EPSG:4326")
    >>> plt.show()
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
    ax.axis('off')

# pylint: disable=protected-access


def _calc_zoom(bbox, coordinate_system):
    import contextily as cx
    bbox = _transform_bbox(bbox, coordinate_system, 'epsg:4326')
    return cx.tile._calculate_zoom(*bbox)


def heatmap(ax, points, weights, bounding_box,
                 map_zoom_level=None,
                 coordinate_system='EPSG:3857',
                 aggregation_type='max',
                 **extra_contextily_params):
    """
    :type ax: AxesSubplot
    :param ax: Matplotlib axes object on which to add the basemap.

    :type points: GeoSeries
    :param points: Sequence of Points

    :type weights: Series(dtype: float|int64)
    :param weights: Weights of point intensity

    :type bounding_box: list
    :param bounding_box: Specify the bounding rectangle [west, south, east, north].

    :type coordinate_system: str
    :param coordinate_system: Coordinate Reference System of the geometry objects.
                    Must be SRID formed, e.g. 'EPSG:4326' or 'EPSG:3857'

    :type aggregation_type: str
    :param aggregation_type: Aggregation type of data processing. Default as 'max'

    :type extra_contextily_params: dict
    :param extra_contextily_params: Extra parameters will be passed to contextily.add_basemap.
                                    See https://contextily.readthedocs.io/en/latest/reference.html for details
    :example:
    >>> import pandas as pd
    >>> import numpy as np
    >>> import arctern
    >>> import matplotlib.pyplot as plt
    >>> # read from test_data.csv
    >>> # Download link: https://raw.githubusercontent.com/arctern-io/arctern-resources/benchmarks/benchmarks/dataset/layer_rendering_test_data/test_data.csv
    >>> df = pd.read_csv("/path/to/test_data.csv", dtype={'longitude':np.float64, 'latitude':np.float64, 'color_weights':np.float64, 'size_weights':np.float64, 'region_boundaries':np.object})
    >>> points = arctern.GeoSeries.point(df['longitude'], df['latitude'])
    >>> 
    >>> # plot heatmap
    >>> fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
    >>> arctern.plot.heatmap(ax, points, df['color_weights'], bounding_box=[-74.01424568752932, 40.72759334104623, -73.96056823889673, 40.76721122683304], coordinate_system='EPSG:4326')
    >>> plt.show()
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
    ax.axis('off')


def choroplethmap(ax, region_boundaries, weights, bounding_box,
                       color_gradient, color_bound=None, opacity=1.0,
                       coordinate_system='EPSG:3857',
                       aggregation_type='max',
                       **extra_contextily_params):
    """
    :type ax: AxesSubplot
    :param ax: Matplotlib axes object on which to add the basemap.

    :type region_boundaries: GeoSeries
    :param region_boundaries: Sequence of polygons, as region boundaries to plot.

    :type weights: Series(dtype: float|int64)
    :param weights: Color weights for polygons

    :type bounding_box: list
    :param bounding_box: Specify the bounding rectangle [west, south, east, north].

    :type color_gradient: list
    :param color_gradient: Specify range of color gradient. 
                           Either use ["hex_color"] to specify a same color for all polygons, 
                           or ["hex_color1", "hex_color2"] to specify a color gradient ranging from "hex_color1" to "hex_color2"
    
    :type color_bound: list
    :param color_bound: Specify weight range [w1, w2] binding to color_gradient. 
                        Needed only when color_gradient has two value ["color1", "color2"]. 
                        Bind w1 to "color1", and w2 to "color2".
                        When weight < w1 or weight > w2, truncate to w1/w2 accordingly. 

    :type opacity: float
    :param opacity: Opacity of polygons, ranged from 0.0 to 1.0, default as 1.0

    :type coordinate_system: str
    :param coordinate_system: Coordinate Reference System of the geometry objects.
                    Must be SRID formed, e.g. 'EPSG:4326' or 'EPSG:3857'

    :type aggregation_type: str
    :param aggregation_type: Aggregation type of data processing. Default as 'max'

    :type extra_contextily_params: dict
    :param extra_contextily_params: Extra parameters will be passed to contextily.add_basemap.
                                    See https://contextily.readthedocs.io/en/latest/reference.html for details
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
    ax.axis('off')


def iconviz(ax, points, bounding_box, icon_path,
                 coordinate_system='EPSG:3857',
                 **extra_contextily_params):
    """
    Plot points as icons on map in Matplotlib
    :type ax: AxesSubplot
    :param ax: Matplotlib axes object on which to add the basemap.

    :type points: GeoSeries
    :param points: Sequence of Points

    :type bounding_box: list
    :param bounding_box: Specify the bounding rectangle [west, south, east, north].

    :type icon_path: str
    :param icon_path: absolute path to icon file

    :type coordinate_system: str
    :param coordinate_system: Coordinate Reference System of the geometry objects.
                    Must be SRID formed, e.g. 'EPSG:4326' or 'EPSG:3857'

    :type extra_contextily_params: dict
    :param extra_contextily_params: Extra parameters will be passed to contextily.add_basemap.
                                    See https://contextily.readthedocs.io/en/latest/reference.html for details
    :example:
    >>> import pandas as pd
    >>> import numpy as np
    >>> import arctern
    >>> import matplotlib.pyplot as plt
    >>> # read from test_data.csv
    >>> # Download link: https://raw.githubusercontent.com/arctern-io/arctern-resources/benchmarks/benchmarks/dataset/layer_rendering_test_data/test_data.csv
    >>> df = pd.read_csv("/path/to/test_data.csv", dtype={'longitude':np.float64, 'latitude':np.float64, 'color_weights':np.float64, 'size_weights':np.float64, 'region_boundaries':np.object}, nrows=10)
    >>> points = arctern.GeoSeries.point(df['longitude'], df['latitude'])
    >>> # plot icon visualization
    >>> # Download icon-viz.png :  https://raw.githubusercontent.com/arctern-io/arctern-docs/master/img/icon/icon-viz.png
    >>> fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
    >>> arctern.plot.iconviz(ax, points, bounding_box=[-74.01424568752932, 40.72759334104623, -73.96056823889673, 40.76721122683304], icon_path='/path/to/icon-viz.png', coordinate_system='EPSG:4326')
    >>> plt.show()
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
    ax.axis('off')

# pylint: disable=too-many-arguments
# pylint: disable=dangerous-default-value


def fishnetmap(ax, points, weights, bounding_box,
                    color_gradient=["#0000FF", "#FF0000"],
                    cell_size=4, cell_spacing=1, opacity=1.0,
                    coordinate_system='epsg:3857',
                    aggregation_type='sum',
                    **extra_contextily_params):
    """
    Plot weighted pointmap in Matplotlib
    :type ax: AxesSubplot
    :param ax: Matplotlib axes object on which to add the basemap.

    :type points: GeoSeries
    :param points: Sequence of Points

    :type weights: Series(dtype: float|int64)
    :param weights: Color weight of points

    :type bounding_box: list
    :param bounding_box: Specify the bounding rectangle [west, south, east, north].

    :type color_gradient: list
    :param color_gradient: Specify range of color gradient. 
                           Either use ["hex_color"] to specify a same color for all points, 
                           or ["hex_color1", "hex_color2"] to specify a color gradient ranging from "hex_color1" to "hex_color2"
                           Current only default value ["#0000FF", "#FF0000"] is supported

    :type cell_size: int
    :param cell_size: Side length of fishnet cells.

    :type cell_spacing: int
    :param cell_spacing: Margin between adjacent fishnet cells.

    :type opacity: float
    :param opacity: Opacity of fishnet, ranged from 0.0 to 1.0, default as 1.0

    :type coordinate_system: str
    :param coordinate_system: Coordinate Reference System of the geometry objects.
                    Must be SRID formed, e.g. 'EPSG:4326' or 'EPSG:3857'

    :type aggregation_type: str
    :param aggregation_type: Aggregation type of data processing. Default as 'sum'

    :type extra_contextily_params: dict
    :param extra_contextily_params: Extra parameters will be passed to contextily.add_basemap
                                    See https://contextily.readthedocs.io/en/latest/reference.html for details
    >>> import pandas as pd
    >>> import numpy as np
    >>> import arctern
    >>> import matplotlib.pyplot as plt
    >>> # read from test_data.csv
    >>> # Download link: https://raw.githubusercontent.com/arctern-io/arctern-resources/benchmarks/benchmarks/dataset/layer_rendering_test_data/test_data.csv
    >>> df = pd.read_csv("/path/to/test_data.csv", dtype={'longitude':np.float64, 'latitude':np.float64, 'color_weights':np.float64, 'size_weights':np.float64, 'region_boundaries':np.object})
    >>> points = arctern.GeoSeries.point(df['longitude'], df['latitude'])
    >>> # render fishnet
    >>> fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
    >>> arctern.plot.fishnetmap(ax, points=points, weights=df['color_weights'], bounding_box=[-74.01424568752932, 40.72759334104623, -73.96056823889673, 40.76721122683304], cell_size=8, cell_spacing=2, opacity=1.0, coordinate_system="EPSG:4326")
    >>> plt.show()
    """
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
    ax.axis('off')
