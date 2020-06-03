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
    "plot_geometry"
    ]

def _flat_polygon(geo_dict, dict_collect):
    if 'polygons' not in dict_collect:
        dict_collect['polygons'] = []
    dict_collect['polygons'].append(geo_dict)

def _flat_line(geo_dict, dict_collect):
    import numpy as np
    if geo_dict['type'] == 'MultiLineString':
        for line in geo_dict['coordinates']:
            line_arry = np.zeros([len(line), 2], dtype=np.double)
            idx = 0
            for coor in line:
                line_arry[idx, 0] = coor[0]
                line_arry[idx, 1] = coor[1]
                idx = idx + 1
            if 'lines' not in dict_collect:
                dict_collect['lines'] = []
            dict_collect['lines'].append(line_arry)
    elif geo_dict['type'] == 'LineString':
        line_arry = np.zeros([len(geo_dict['coordinates']), 2], dtype=np.double)
        idx = 0
        for coor in geo_dict['coordinates']:
            line_arry[idx, 0] = coor[0]
            line_arry[idx, 1] = coor[1]
            idx = idx + 1
        if 'lines' not in dict_collect:
            dict_collect['lines'] = []
        dict_collect['lines'].append(line_arry)

def _flat_point(geo_dict, dict_collect):
    if geo_dict['type'] == 'Point':
        if 'points' not in dict_collect:
            dict_collect['points'] = []
        dict_collect['points'].append(geo_dict['coordinates'])
    elif geo_dict['type'] == 'MultiPoint':
        if 'points' not in dict_collect:
            dict_collect['points'] = []
        dict_collect['points'].extend(geo_dict['coordinates'])

def _flat_geoms(geo_dict, dict_collect):
    if geo_dict['type'] == 'GeometryCollection':
        for geos in geo_dict['geometries']:
            _flat_geoms(geos, dict_collect)
    elif geo_dict['type'] == 'MultiPolygon' or geo_dict['type'] == 'Polygon':
        _flat_polygon(geo_dict, dict_collect)
    elif geo_dict['type'] == 'MultiLineString' or geo_dict['type'] == 'LineString':
        _flat_line(geo_dict, dict_collect)
    elif geo_dict['type'] == 'Point' or geo_dict['type'] == 'MultiPoint':
        _flat_point(geo_dict, dict_collect)
    else:
        raise RuntimeError(f"unsupported geometry: {geo_dict}")

def _plot_polygons(ax, polygons, **style_kwds):
    try:
        from descartes.patch import PolygonPatch
    except ImportError:
        raise ImportError(
            "The descartes package is required for plotting polygons in arctern. "
            "You can install it using 'conda install -c conda-forge descartes' ")
    try:
        from matplotlib.collections import PatchCollection
    except ImportError:
        raise ImportError(
            "The matplotlib package is required for plotting polygons in arctern. "
            "You can install it using 'conda install -c conda-forge matplotlib' ")
    collection = PatchCollection([PolygonPatch(geo) for geo in polygons], **style_kwds)
    ax.add_collection(collection, autolim=True)

# value for linestyles : solid|dashed|dashdot|dotted
def _plot_lines(ax, lines, **style_kwds):
    try:
        from matplotlib.collections import LineCollection
        import matplotlib as mpl
    except ImportError:
        raise ImportError(
            "The matplotlib package is required for plotting polygons in arctern. "
            "You can install it using 'conda install -c conda-forge matplotlib' ")

    collection = LineCollection(lines, **style_kwds)
    ax.add_collection(collection, autolim=True)

def _plot_points(ax, x, y, **style_kwds):
    if 'markersize' in style_kwds:
        style_kwds['s'] = style_kwds['markersize']
        del style_kwds['markersize']
    ax.scatter(x, y, **style_kwds)

def _get_random_color_from_cycle():
    import random
    try:
        import matplotlib as mpl
    except ImportError:
        raise ImportError(
            "The matplotlib package is required for plotting polygons in arctern. "
            "You can install it using 'conda install -c conda-forge matplotlib' ")

    cycle_list = mpl.rcParams['axes.prop_cycle'].by_key()['color']
    cyc_idx = random.randrange(0, len(cycle_list))
    return cycle_list[cyc_idx]

# pylint: disable=too-many-return-statements
# pylint: disable=too-many-branches
def _get_style_value(geo_name, style_key, style_vale):
    try:
        from matplotlib.colors import is_color_like
        import matplotlib as mpl
    except ImportError:
        raise ImportError(
            "The matplotlib package is required for plotting polygons in arctern. "
            "You can install it using 'conda install -c conda-forge matplotlib' ")

    if style_key == 'alpha':
        return style_vale if style_vale is not None else 1

    if geo_name == 'polygons':
        if style_key == 'linewidth':
            return style_vale if style_vale is not None else mpl.rcParams['patch.linewidth']
        if style_key == 'linestyle':
            return style_vale if style_vale is not None else mpl.rcParams['lines.linestyle']
        if style_key == 'edgecolor':
            return style_vale if is_color_like(style_vale) else mpl.rcParams['patch.edgecolor']
        if style_key == 'facecolor':
            return style_vale if style_vale is not None else mpl.rcParams['patch.facecolor']
    elif geo_name == 'lines':
        if style_key == 'color':
            return style_vale if is_color_like(style_vale) else mpl.rcParams['lines.color']
        if style_key == 'linewidth':
            return style_vale if style_vale is not None else mpl.rcParams['lines.linewidth']
        if style_key == 'linestyle':
            return style_vale if style_vale is not None else mpl.rcParams['lines.linestyle']
    elif geo_name == 'points':
        if style_key == 'color':
            return style_vale if is_color_like(style_vale) else _get_random_color_from_cycle()
        if style_key == 'marker':
            return style_vale if style_vale is not None else mpl.rcParams['scatter.marker']
        if style_key == 'markersize':
            return style_vale if style_vale is not None else mpl.rcParams['lines.markersize']
    return None

def _extend_collect(geo_name, geo_collect, plot_collect, row_style, geo_style):
    if geo_name in geo_collect:
        if geo_name not in plot_collect:
            plot_collect[geo_name] = []
        plot_collect[geo_name].extend(geo_collect[geo_name])

        for style_key, style_val in row_style.items():
            if style_key not in geo_style:
                geo_style[style_key] = []
            style_val = _get_style_value(geo_name, style_key, style_val)
            if style_val is None:
                del geo_style[style_key]
            else:
                style = [style_val for _ in range(len(geo_collect[geo_name]))]
                geo_style[style_key].extend(style)

def _add_global_plot_style(geo_name, style_key, style_val, plot_style):
    value = _get_style_value(geo_name, style_key, style_val)
    if value is not None:
        plot_style[style_key] = value


def _plot_collection(ax, geoms_list, **style_kwds):
    import json
    style_iter = dict()
    polygons_style = dict()
    lines_style = dict()
    points_style = dict()
    for key, val in style_kwds.items():
        if isinstance(val, (str, int, float)):
            _add_global_plot_style('polygons', key, val, polygons_style)
            _add_global_plot_style('lines', key, val, lines_style)
            _add_global_plot_style('points', key, val, points_style)
        else:
            try:
                style_iter[key] = iter(val)
            except TypeError:
                _add_global_plot_style('polygons', key, val, polygons_style)
                _add_global_plot_style('lines', key, val, lines_style)
                _add_global_plot_style('points', key, val, points_style)

    plot_collect = dict()
    for geo in geoms_list:
        row_style = dict()
        for key, val_iter in style_iter.items():
            val = next(val_iter, None)
            row_style[key] = val

        if geo is not None:
            geo_dict = json.loads(geo)
            geo_collect = dict()
            _flat_geoms(geo_dict, geo_collect)
            _extend_collect('polygons', geo_collect, plot_collect, row_style, polygons_style)
            _extend_collect('lines', geo_collect, plot_collect, row_style, lines_style)
            _extend_collect('points', geo_collect, plot_collect, row_style, points_style)

    if 'polygons' in plot_collect:
        _plot_polygons(ax, plot_collect['polygons'], **polygons_style)
    if 'lines' in plot_collect:
        _plot_lines(ax, plot_collect['lines'], **lines_style)
    if 'points' in plot_collect:
        x = [p[0] for p in plot_collect['points']]
        y = [p[1] for p in plot_collect['points']]
        _plot_points(ax, x, y, **points_style)
    ax.autoscale_view()

def _plot_pandas_series(ax, geoms, **style_kwds):
    import pandas.core.series
    import arctern

    if not isinstance(geoms, pandas.core.series.Series):
        raise TypeError("geoms shuld be type of pandas.core.series.Series")
    len_geoms = len(geoms)
    if len_geoms < 1:
        return None
    if isinstance(geoms[0], str):
        pass
    elif isinstance(geoms[0], bytes):
        geoms = arctern.ST_AsGeoJSON(geoms)
    else:
        raise RuntimeError(f"unexpected input type, {type(geoms[0])}")

    _plot_collection(ax, geoms, **style_kwds)
    return None

def plot_geometry(ax, geoms, **style_kwds):
    """
    Plot a collection of geometries to `ax`. Parameters 'linewidth', 'linestyle', 'edgecolor',
    'facecolor', 'color', 'marker', 'markersize' are used to describe the style of plotted figure.

    For geometry types `Polygon` and `MultiPolygon`, only 'linewidth', 'linestyle', 'edgecolor',
    'facecolor' are effective.

    For geometry types `Linestring` and `MultiLinestring`, only 'color', 'linewidth', 'linestyle' are effective.

    For geometry types `Point` and `MultiPoint`, only 'color', 'marker', 'markersize' are effective.

    :type ax: matplotlib.axes.Axes
    :param ax: The axes where geometries will be plotted.

    :type geoms: Series or DataFrame
    :param geoms: sequence of geometries.

    :type linewidth: list(float)
    :param linewidth: The width of line, the default value is 1.0.

    :type linestyle: list(string)
    :param linestyle: The style of the line， the default value is '-'.

    :type edgecolor: list(string)
    :param edgecolor: The edge color of the geometry, the default value is 'black'.

    :type facecolor: list(string)
    :param facecolor: The color of the face of the geometry, the default value is 'C0'.

    :type color: list(string)
    :param color: The color of the geometry, the default value is 'C0'.

    :type marker: string
    :param marker: The shape of point, the default value is 'o'.

    :type markersize: double
    :param markersize: The size of points, the default value is 6.0.

    :type alpha: double
    :param alpha: The transparency of the geometry, the default value is 1.0.


    :example:
       >>> import pandas
       >>> import matplotlib.pyplot as plt
       >>> import arctern
       >>> raw_data = []
       >>> raw_data.append('point(0 0)')
       >>> raw_data.append('linestring(0 10, 5 5, 10 0)')
       >>> raw_data.append('polygon((2 2,2 3,3 3,3 2,2 2))')
       >>> raw_data.append("GEOMETRYCOLLECTION("
                           "polygon((1 1,1 2,2 2,2 1,1 1)),"
                           "linestring(0 1, 5 6, 10 11),"
                           "POINT(4 7))")
       >>> arr_wkt = pandas.Series(raw_data)
       >>> arr_wkb = arctern.ST_CurveToLine(arctern.ST_GeomFromText(arr_wkt))
       >>> df = pandas.DataFrame({'wkb':arr_wkb})
       >>> fig, ax = plt.subplots()
       >>> arctern.plot.plot_geometry(ax, df,
                                 color=['orange', 'green', 'blue', 'red'],
                                 marker='^',
                                 markersize=100,
                                 linewidth=[None, 7, 8, 5],
                                 linestyle=[None, 'dashed', 'dashdot', None],
                                 edgecolor=[None, None, 'red', None],
                                 facecolor=[None, None, 'black', None])
       >>> ax.grid()
       >>> fig.savefig('/tmp/plot_test.png')
    """
    import pandas.core.series
    if isinstance(geoms, pandas.core.series.Series):
        _plot_pandas_series(ax, geoms, **style_kwds)
    elif isinstance(geoms, pandas.core.frame.DataFrame):
        if len(geoms.columns) != 1:
            raise RuntimeError(f"The input param 'geoms' should have only one column. geoms schema = {geoms.columns}")
        geom_series = geoms[geoms. columns[0]]
        _plot_pandas_series(ax, geom_series, **style_kwds)
