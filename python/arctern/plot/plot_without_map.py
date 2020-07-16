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

from arctern.util.vega import vega_pointmap, vega_weighted_pointmap, vega_heatmap, vega_choroplethmap, vega_icon, vega_fishnetmap, vega_unique_value_choroplethmap
import arctern


def pointmap_layer(w, h, points, bounding_box=None,
                  point_size=3, point_color='#115f9a', opacity=1.0,
                  coordinate_system='EPSG:3857'):
    """
    Plots a point map layer.

    Parameters
    ----------
    w : int
        Width of the output PNG image.
    h : int
        Height of the output PNG image.
    points : GeoSeries
        Sequence of points.
    bounding_box : list
        Bounding box of the map. For example, [west, south, east, north].
    point_size : int, optional
        Diameter of points, by default 3.
    point_color : str, optional
        Point color in Hex Color Code, by default '#115f9a'.
    opacity : float, optional
        Opacity of points, ranged from 0.0 to 1.0, by default 1.0.
    coordinate_system : str, optional
        The Coordinate Reference System (CRS) set to all geometries, by default 'EPSG:3857'.
        Only supports SRID as a WKT representation of CRS by now.

    Returns
    -------
    bytes
        A base64 encoded PNG image.

    Examples
    ---------

    .. plot::
       :context: close-figs

       >>> import pandas as pd
       >>> import numpy as np
       >>> import arctern
       >>> import matplotlib.pyplot as plt
       >>> import io
       >>> import base64
       >>> # Read from test_data.csv
       >>> # Download link: https://raw.githubusercontent.com/arctern-io/arctern-resources/benchmarks/benchmarks/dataset/layer_rendering_test_data/test_data.csv
       >>> # Uncomment the lines below to download the test data
       >>> # import os
       >>> # os.system('wget "https://raw.githubusercontent.com/arctern-io/arctern-resources/benchmarks/benchmarks/dataset/layer_rendering_test_data/test_data.csv"')
       >>> df = pd.read_csv(filepath_or_buffer="test_data.csv", dtype={'longitude':np.float64, 'latitude':np.float64, 'color_weights':np.float64, 'size_weights':np.float64, 'region_boundaries':np.object})
       >>> points = arctern.GeoSeries.point(df['longitude'], df['latitude'])
       >>>
       >>> # Plot pointmap_layer
       >>> bbox=[-73.99668712186558,40.72972339069935,-73.99045479584949,40.7345193345495]
       >>> map_layer=arctern.plot.pointmap_layer(1024, 896, points, bounding_box=bbox, point_size=10, point_color="#115f9a", opacity=1.0, coordinate_system="EPSG:4326")
       >>> fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
       >>> f = io.BytesIO(base64.b64decode(map_layer))
       >>> img = plt.imread(f)
       >>> ax.imshow(img)
       >>> ax.axis('off')
       >>> plt.show()
    """
    vega = vega_pointmap(w, h, bounding_box=bounding_box, point_size=point_size,
                         point_color=point_color, opacity=opacity, coordinate_system=coordinate_system)
    hexstr = arctern.point_map_layer(vega, points)
    return hexstr


# pylint: disable=too-many-arguments
# pylint: disable=dangerous-default-value
def weighted_pointmap_layer(w, h, points, color_weights=None,
                           size_weights=None,
                           bounding_box=None,
                           color_gradient=["#115f9a", "#d0f400"],
                           color_bound=[0, 0],
                           size_bound=[3],
                           opacity=1.0,
                           coordinate_system='EPSG:3857'):
    """
    Plots a weighted point map layer.

    Parameters
    ----------
    w : int
        Width of the output PNG image.
    h : int
        Height of the output PNG image.
    points : GeoSeries
        Sequence of points.
    color_weights : Series, optional
        Weights of point color.
    size_weights : Series, optional
        Weights of point size.
    bounding_box : list
        Bounding box of the map. For example, [west, south, east, north].
    color_gradient : list, optional
        Range of color gradient, by default ["#115f9a", "#d0f400"].
        Either use ["hex_color"] to specify a same color for all geometries, or ["hex_color1", "hex_color2"] to specify a color gradient ranging from "hex_color1" to "hex_color2".
    color_bound : list, optional
        Weight range [w1, w2] of ``color_gradient``, by default [0, 0].
        Needed only when ``color_gradient`` has two values ["color1", "color2"]. Binds w1 to "color1", and w2 to "color2". When weight < w1 or weight > w2, the weight will be truncated to w1 or w2 accordingly.
    size_bound : list, optional
        Weight range [w1, w2] of ``size_weights``, by default [3]. When weight < w1 or weight > w2, the weight will be truncated to w1 or w2 accordingly.
    opacity : float, optional
        Opacity of points, ranged from 0.0 to 1.0, by default 1.0.
    coordinate_system : str, optional
        The Coordinate Reference System (CRS) set to all geometries, by default 'EPSG:3857'.
        Only supports SRID as a WKT representation of CRS by now.

    Returns
    -------
    bytes
        A base64 encoded PNG image.

    Examples
    -------

    .. plot::
       :context: close-figs

       >>> import pandas as pd
       >>> import numpy as np
       >>> import arctern
       >>> import matplotlib.pyplot as plt
       >>> import io
       >>> import base64
       >>> # Read from test_data.csv
       >>> # Download link: https://raw.githubusercontent.com/arctern-io/arctern-resources/benchmarks/benchmarks/dataset/layer_rendering_test_data/test_data.csv
       >>> # Uncomment the lines below to download the test data
       >>> # import os
       >>> # os.system('wget "https://raw.githubusercontent.com/arctern-io/arctern-resources/benchmarks/benchmarks/dataset/layer_rendering_test_data/test_data.csv"')
       >>> df = pd.read_csv(filepath_or_buffer="test_data.csv", dtype={'longitude':np.float64, 'latitude':np.float64, 'color_weights':np.float64, 'size_weights':np.float64, 'region_boundaries':np.object})
       >>> points = arctern.GeoSeries.point(df['longitude'], df['latitude'])
       >>>
       >>> # Plot weighted_pointmap_layer
       >>> bbox=[-73.99668712186558,40.72972339069935,-73.99045479584949,40.7345193345495]
       >>> map_layer = arctern.plot.weighted_pointmap_layer(1024, 896, points, color_weights=df['color_weights'], bounding_box=bbox, color_gradient=["#115f9a", "#d0f400"], color_bound=[2.5,15], size_bound=[16], opacity=1.0, coordinate_system="EPSG:4326")
       >>> fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
       >>> f = io.BytesIO(base64.b64decode(map_layer))
       >>> img = plt.imread(f)
       >>> ax.imshow(img)
       >>> ax.axis('off')
       >>> plt.show()
    """
    vega = vega_weighted_pointmap(w, h, bounding_box=bounding_box, color_gradient=color_gradient,
                                  color_bound=color_bound, size_bound=size_bound, opacity=opacity,
                                  coordinate_system=coordinate_system)
    hexstr = arctern.weighted_point_map_layer(
        vega, points, color_weights=color_weights, size_weights=size_weights)
    return hexstr


# pylint: disable=protected-access
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


# pylint: disable=protected-access
def _calc_zoom(bbox, coordinate_system):
    import contextily as cx
    bbox = _transform_bbox(bbox, coordinate_system, 'epsg:4326')
    return cx.tile._calculate_zoom(*bbox)


def heatmap_layer(w, h, points, weights, bounding_box,
                 map_zoom_level=None,
                 coordinate_system='EPSG:3857',
                 aggregation_type='max'):
    """
    Plots a heat map layer.

    Parameters
    ----------
    w : int
        Width of the output PNG image.
    h : int
        Height of the output PNG image.
    points : GeoSeries
        Sequence of points.
    weights : Series
        Weights of point intensity.
    bounding_box : list
        Bounding box of the map. For example, [west, south, east, north].
    map_zoom_level : [type], optional
        Zoom level of the map, by default 'auto'.
    coordinate_system : str, optional
        The Coordinate Reference System (CRS) set to all geometries, by default 'EPSG:3857'.
        Only supports SRID as a WKT representation of CRS by now.
    aggregation_type : str, optional
        Aggregation type, by default 'max'.

    Returns
    -------
    bytes
        A base64 encoded PNG image.

    Examples
    -------

    .. plot::
       :context: close-figs

       >>> import pandas as pd
       >>> import numpy as np
       >>> import arctern
       >>> import matplotlib.pyplot as plt
       >>> import io
       >>> import base64
       >>> # Read from test_data.csv
       >>> # Download link: https://raw.githubusercontent.com/arctern-io/arctern-resources/benchmarks/benchmarks/dataset/layer_rendering_test_data/test_data.csv
       >>> # Uncomment the lines below to download the test data
       >>> # import os
       >>> # os.system('wget "https://raw.githubusercontent.com/arctern-io/arctern-resources/benchmarks/benchmarks/dataset/layer_rendering_test_data/test_data.csv"')
       >>> df = pd.read_csv(filepath_or_buffer="test_data.csv", dtype={'longitude':np.float64, 'latitude':np.float64, 'color_weights':np.float64, 'size_weights':np.float64, 'region_boundaries':np.object})
       >>> points = arctern.GeoSeries.point(df['longitude'], df['latitude'])
       >>>
       >>> # Plot heatmap_layer
       >>> bbox = [-74.01424568752932, 40.72759334104623, -73.96056823889673, 40.76721122683304]
       >>> map_layer = arctern.plot.heatmap_layer(1024, 896, points, df['color_weights'], bounding_box=bbox, coordinate_system='EPSG:4326')
       >>> fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
       >>> f = io.BytesIO(base64.b64decode(map_layer))
       >>> img = plt.imread(f)
       >>> ax.imshow(img)
       >>> ax.axis('off')
       >>> plt.show()
    """
    if map_zoom_level is None:
        map_zoom_level = _calc_zoom(bounding_box, coordinate_system)
    vega = vega_heatmap(w, h, bounding_box=bounding_box, map_zoom_level=map_zoom_level,
                        aggregation_type=aggregation_type, coordinate_system=coordinate_system)
    hexstr = arctern.heat_map_layer(vega, points, weights)
    return hexstr


def choroplethmap_layer(w, h, region_boundaries, weights, bounding_box,
                       color_gradient, color_bound=None, opacity=1.0,
                       coordinate_system='EPSG:3857',
                       aggregation_type='max'):
    """
    Plots a choropleth map layer.

    Parameters
    ----------
    w : int
        Width of the output PNG image.
    h : int
        Height of the output PNG image.
    region_boundaries : GeoSeries
        Sequence of polygons, as region boundaries to plot.
    weights : Series
        Color weights for polygons
    bounding_box : list
        Bounding box of the map. For example, [west, south, east, north].
    color_gradient : list
        Range of color gradient.
        Either use ["hex_color"] to specify a same color for all geometries, or ["hex_color1", "hex_color2"] to specify a color gradient ranging from "hex_color1" to "hex_color2".
    color_bound : list, optional
        Weight range [w1, w2] of ``color_gradient``.
        Needed only when ``color_gradient`` has two values ["color1", "color2"]. Binds w1 to "color1", and w2 to "color2". When weight < w1 or weight > w2, the weight will be truncated to w1 or w2 accordingly.
    opacity : float, optional
        Opacity of polygons, ranged from 0.0 to 1.0, by default 1.0.
    coordinate_system : str, optional
        The Coordinate Reference System (CRS) set to all geometries, by default 'EPSG:3857'.
        Only supports SRID as a WKT representation of CRS by now.
    aggregation_type : str, optional
        Aggregation type, by default 'max'.

    Returns
    -------
    bytes
        A base64 encoded PNG image.

    Examples
    -------

    .. plot::
       :context: close-figs

       >>> import pandas as pd
       >>> import numpy as np
       >>> import arctern
       >>> import matplotlib.pyplot as plt
       >>> import io
       >>> import base64
       >>> # Read from test_data.csv
       >>> # Download link: https://raw.githubusercontent.com/arctern-io/arctern-resources/benchmarks/benchmarks/dataset/layer_rendering_test_data/test_data.csv
       >>> # Uncomment the lines below to download the test data
       >>> # import os
       >>> # os.system('wget "https://raw.githubusercontent.com/arctern-io/arctern-resources/benchmarks/benchmarks/dataset/layer_rendering_test_data/test_data.csv"')
       >>> df = pd.read_csv(filepath_or_buffer="test_data.csv", dtype={'longitude':np.float64, 'latitude':np.float64, 'color_weights':np.float64, 'size_weights':np.float64, 'region_boundaries':np.object})
       >>> input = df[pd.notna(df['region_boundaries'])].groupby(['region_boundaries']).mean().reset_index()
       >>> polygon = arctern.GeoSeries(input['region_boundaries'])
       >>>
       >>> # Plot choroplethmap layer
       >>> bbox = [-74.01124953254566,40.73413446570038,-73.96238859103838,40.766161712662296]
       >>> map_layer = arctern.plot.choroplethmap_layer(1024, 896, polygon, input['color_weights'], bounding_box=bbox, color_gradient=["#115f9a","#d0f400"], color_bound=[5,18], opacity=1.0, coordinate_system='EPSG:4326', aggregation_type="mean")
       >>> fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
       >>> f = io.BytesIO(base64.b64decode(map_layer))
       >>> img = plt.imread(f)
       >>> ax.imshow(img)
       >>> ax.axis('off')
       >>> plt.show()
    """
    vega = vega_choroplethmap(w, h, bounding_box=bounding_box, color_gradient=color_gradient, color_bound=color_bound,
                              opacity=opacity, aggregation_type=aggregation_type, coordinate_system=coordinate_system)
    hexstr = arctern.choropleth_map_layer(vega, region_boundaries, weights)
    return hexstr


def iconviz_layer(w, h, points, bounding_box, icon_path,
                 icon_size=None, coordinate_system='EPSG:3857'):
    """
    Plots an icon map layer.

    Parameters
    ----------
    w : int
        Width of the output PNG image.
    h : int
        Height of the output PNG image.
    points : GeoSeries
        Sequence of points.
    bounding_box : list
        Bounding box of the map. For example, [west, south, east, north].
    icon_path : str
        Absolute path to icon file.
    icon_size : list
        Size of the icon, a list with width and height of icon. For example, [width, height].
    coordinate_system : str, optional
        The Coordinate Reference System (CRS) set to all geometries, by default 'EPSG:3857'.
        Only supports SRID as a WKT representation of CRS by now.

    Returns
    -------
    bytes
        A base64 encoded PNG image.

    Examples
    -------

    .. plot::
       :context: close-figs

       >>> import pandas as pd
       >>> import numpy as np
       >>> import arctern
       >>> import matplotlib.pyplot as plt
       >>> import io
       >>> import base64
       >>> # Read from test_data.csv
       >>> # Download link: https://raw.githubusercontent.com/arctern-io/arctern-resources/benchmarks/benchmarks/dataset/layer_rendering_test_data/test_data.csv
       >>> # Uncomment the lines below to download the test data
       >>> # import os
       >>> # os.system('wget "https://raw.githubusercontent.com/arctern-io/arctern-resources/benchmarks/benchmarks/dataset/layer_rendering_test_data/test_data.csv"')
       >>> df = pd.read_csv(filepath_or_buffer="test_data.csv", dtype={'longitude':np.float64, 'latitude':np.float64, 'color_weights':np.float64, 'size_weights':np.float64, 'region_boundaries':np.object})
       >>> points = arctern.GeoSeries.point(df['longitude'], df['latitude'])
       >>>
       >>> # Plot icon visualization
       >>> # Download icon-viz.png :  https://raw.githubusercontent.com/arctern-io/arctern-docs/master/img/icon/icon-viz.png
       >>> # Uncomment the line below to download the icon image
       >>> # os.system('wget "https://raw.githubusercontent.com/arctern-io/arctern-docs/master/img/icon/icon-viz.png"')
       >>> bbox = [-74.01424568752932, 40.72759334104623, -73.96056823889673, 40.76721122683304]
       >>> map_layer = arctern.plot.iconviz_layer(1024, 896, points, bounding_box=bbox, icon_path='icon-viz.png', coordinate_system='EPSG:4326')
       >>> fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
       >>> f = io.BytesIO(base64.b64decode(map_layer))
       >>> img = plt.imread(f)
       >>> ax.imshow(img)
       >>> ax.axis('off')
       >>> plt.show()
    """
    vega = vega_icon(w, h, bounding_box=bounding_box, icon_path=icon_path, icon_size=icon_size,
                     coordinate_system=coordinate_system)
    hexstr = arctern.icon_viz_layer(vega, points)
    return hexstr


# pylint: disable=too-many-arguments
# pylint: disable=dangerous-default-value
def fishnetmap_layer(w, h, points, weights, bounding_box,
                    color_gradient=["#0000FF", "#FF0000"],
                    cell_size=4, cell_spacing=1, opacity=1.0,
                    coordinate_system='epsg:3857',
                    aggregation_type='sum'):
    """
    Plots a fishnet map layer.

    Parameters
    ----------
    w : int
        Width of the output PNG image.
    h : int
        Height of the output PNG image.
    points : GeoSeries
        Sequence of points.
    weights : Series
        Color weights of polygons.
    bounding_box : list
        Bounding box of the map. For example, [west, south, east, north].
    color_gradient : list, optional
        Range of color gradient, by default ["#0000FF", "#FF0000"].
        Either use ["hex_color"] to specify a same color for all geometries, or ["hex_color1", "hex_color2"] to specify a color gradient ranging from "hex_color1" to "hex_color2".
    cell_size : int, optional
        Side length of fishnet cells, by default 4.
    cell_spacing : int, optional
        Margin between adjacent fishnet cells, by default 1.
    opacity : float, optional
        Opacity of the fishnet, ranged from 0.0 to 1.0, by default 1.0.
    coordinate_system : str, optional
        The Coordinate Reference System (CRS) set to all geometries, by default 'EPSG:3857'.
        Only supports SRID as a WKT representation of CRS by now.
    aggregation_type : str, optional
        Aggregation type, by default 'sum'.

    Returns
    -------
    bytes
        A base64 encoded PNG image.

    Examples
    -------

    .. plot::
       :context: close-figs

       >>> import pandas as pd
       >>> import numpy as np
       >>> import arctern
       >>> import matplotlib.pyplot as plt
       >>> import io
       >>> import base64
       >>> # Read from test_data.csv
       >>> # Download link: https://raw.githubusercontent.com/arctern-io/arctern-resources/benchmarks/benchmarks/dataset/layer_rendering_test_data/test_data.csv
       >>> # Uncomment the lines below to download the test data
       >>> # import os
       >>> # os.system('wget "https://raw.githubusercontent.com/arctern-io/arctern-resources/benchmarks/benchmarks/dataset/layer_rendering_test_data/test_data.csv"')
       >>> df = pd.read_csv(filepath_or_buffer="test_data.csv", dtype={'longitude':np.float64, 'latitude':np.float64, 'color_weights':np.float64, 'size_weights':np.float64, 'region_boundaries':np.object})
       >>> points = arctern.GeoSeries.point(df['longitude'], df['latitude'])
       >>>
       >>> # render fishnet layer
       >>> bbox = [-74.01424568752932, 40.72759334104623, -73.96056823889673, 40.76721122683304]
       >>> map_layer = arctern.plot.fishnetmap_layer(1024, 896, points=points, weights=df['color_weights'], bounding_box=bbox, cell_size=8, cell_spacing=2, opacity=1.0, coordinate_system="EPSG:4326")
       >>> fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
       >>> f = io.BytesIO(base64.b64decode(map_layer))
       >>> img = plt.imread(f)
       >>> ax.imshow(img)
       >>> ax.axis('off')
       >>> plt.show()
    """
    vega = vega_fishnetmap(w, h, bounding_box=bounding_box, color_gradient=color_gradient,
                           cell_size=cell_size, cell_spacing=cell_spacing, opacity=opacity,
                           coordinate_system=coordinate_system, aggregation_type=aggregation_type)
    hexstr = arctern.fishnet_map_layer(vega, points, weights)
    return hexstr


def unique_value_choropleth_map_layer(w, h, region_boundaries, labels, bounding_box,
                                     unique_value_infos={}, opacity=1.0,
                                     coordinate_system='EPSG:3857'):
    """
    Plots a choropleth map layer.

    Parameters
    ----------
    w : int
        Width of the output PNG image.
    h : int
        Height of the output PNG image.
    region_boundaries : GeoSeries
        Sequence of polygons, as region boundaries to plot.
    labels : Series
        Color labels for polygons
    bounding_box : list
        Bounding box of the map. For example, [west, south, east, north].
    unique_value_infos : dict
        key-value pairs, key represents the label, and value represents the color corresponding to the label.
    opacity : float, optional
        Opacity of polygons, ranged from 0.0 to 1.0, by default 1.0.
    coordinate_system : str, optional
        The Coordinate Reference System (CRS) set to all geometries, by default 'EPSG:3857'.
        Only supports SRID as a WKT representation of CRS by now.

    Returns
    -------
    bytes
        A base64 encoded PNG image.

    Examples
    -------

    .. plot::
       :context: close-figs

       >>> import pandas as pd
       >>> import numpy as np
       >>> import arctern
       >>> import matplotlib.pyplot as plt
       >>> import io
       >>> import base64
       >>> # Read from test_data.csv
       >>> # Download link: https://raw.githubusercontent.com/arctern-io/arctern-resources/benchmarks/benchmarks/dataset/layer_rendering_test_data/test_data.csv
       >>> # Uncomment the lines below to download the test data
       >>> # import os
       >>> # os.system('wget "https://raw.githubusercontent.com/arctern-io/arctern-resources/benchmarks/benchmarks/dataset/layer_rendering_test_data/test_data.csv"')
       >>> df = pd.read_csv(filepath_or_buffer="test_data.csv", dtype={'longitude':np.float64, 'latitude':np.float64, 'color_weights':np.float64, 'size_weights':np.float64, 'region_boundaries':np.object})
       >>> input = df[pd.notna(df['region_boundaries'])].groupby(['region_boundaries']).mean().reset_index()
       >>> polygon = arctern.GeoSeries(input['region_boundaries'])
       >>>
       >>> # Plot choroplethmap layer
       >>> bbox = [-74.01124953254566,40.73413446570038,-73.96238859103838,40.766161712662296]
    """
    vega = vega_unique_value_choroplethmap(w, h, bounding_box=bounding_box, unique_value_infos=unique_value_infos,
                              opacity=opacity, coordinate_system=coordinate_system)
    hexstr = arctern.unique_value_choropleth_map_layer(vega, region_boundaries, labels)
    return hexstr
