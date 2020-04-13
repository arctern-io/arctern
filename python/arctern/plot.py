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
    import json
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

def _get_attr(attr_list, **style_kwds):
    attr_val = dict()
    for attr in attr_list:
        if attr in style_kwds:
            attr_val[attr] = style_kwds[attr]
    return attr_val

def _plot_polygons(ax, polygons, **style_kwds):
    try:
        from descartes.patch import PolygonPatch
    except ImportError:
        raise ImportError(
            "The descartes package is required for plotting polygons in geopandas. "
            "You can install it using 'conda install -c conda-forge descartes' "
        )
    try:
        from matplotlib.collections import PatchCollection
    except ImportError:
        raise ImportError(
            "The matplotlib package is required for plotting polygons in geopandas. "
            "You can install it using 'conda install -c conda-forge descartes' "
        )
    attr = _get_attr(['linewidth', 'linestyle', 'edgecolor', 'facecolor'], **style_kwds)
    collection = PatchCollection([PolygonPatch(geo) for geo in polygons], **attr)
    ax.add_collection(collection, autolim=True)

# value for linestyles : solid|dashed|dashdot|dotted
def _plot_lines(ax, lines, **style_kwds):
    try:
        from matplotlib.collections import LineCollection
    except ImportError:
        raise ImportError(
            "The matplotlib package is required for plotting polygons in geopandas. "
            "You can install it using 'conda install -c conda-forge descartes' "
        )
    attr = _get_attr(['color', 'linewidth', 'linestyle'], **style_kwds)
    collection = LineCollection(lines, **attr)
    ax.add_collection(collection, autolim=True)

def _plot_points(ax, x, y, **style_kwds):
    attr = _get_attr(['color', 'marker'], **style_kwds)
    if 'markersize' in style_kwds:
        attr['s'] = style_kwds['markersize']
    ax.scatter(x, y, **attr)

def _extend_collect(geo_name, geo_collect, plot_collect, row_style, geo_style):
    if geo_name in geo_collect:
        if geo_name not in plot_collect:
            plot_collect[geo_name] = []
        plot_collect[geo_name].extend(geo_collect[geo_name])

        for style_key, style_val in row_style.items():
            if style_key not in geo_style:
                geo_style[style_key] = []
            style = [style_val for _ in range(len(geo_collect[geo_name]))]
            geo_style[style_key].extend(style)

def _plot_collection(ax, geoms_list, **style_kwds):
    import json

    style_iter = dict()
    polygons_style = dict()
    lines_style = dict()
    points_style = dict()
    for key, val in style_kwds.items():
        try:
            style_iter[key] = iter(val)
        except TypeError:
            polygons_style[key] = val
            lines_style[key] = val
            points_style[key] = val

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
        _plot_polygons(ax, plot_collect['polygons'], **style_kwds)
    if 'lines' in plot_collect:
        _plot_lines(ax, plot_collect['lines'], **style_kwds)
    if 'points' in plot_collect:
        x = [p[0] for p in plot_collect['points']]
        y = [p[1] for p in plot_collect['points']]
        _plot_points(ax, x, y, **style_kwds)
    ax.autoscale_view()

def _plot_pandas_series(ax, geoms, **style_kwds):
    import pandas.core.series

    if not isinstance(geoms, pandas.core.series.Series):
        raise TypeError("geoms shuld be type of pandas.core.series.Series")
    len_geoms = len(geoms)
    if len_geoms < 1:
        return None
    if isinstance(geoms[0], str):
        pass
    elif isinstance(geoms[0], bytes):
        import arctern
        geoms = arctern.ST_AsGeoJSON(geoms)
    else:
        raise RuntimeError(f"unexpected input type, {type(geoms[0])}")

    _plot_collection(ax, geoms, **style_kwds)
    return None

def plot(ax, geoms, **style_kwds):
    import pandas.core.series
    if isinstance(geoms, pandas.core.series.Series):
        _plot_pandas_series(ax, geoms, **style_kwds)
    elif isinstance(geoms, pandas.core.frame.DataFrame):
        if len(geoms.columns) != 1:
            raise RuntimeError(f"The input param 'geoms' should have only one column. geoms schema = {geoms.columns}")
        geom_series = geoms[geoms. columns[0]]
        _plot_pandas_series(ax, geom_series, **style_kwds)
