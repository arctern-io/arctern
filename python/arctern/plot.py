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
            line_arry = np.zeros([len(line), 2])
            idx = 0
            for coor in line:
                line_arry[idx, 0] = coor[0]
                line_arry[idx, 1] = coor[1]
                idx = idx + 1
            if 'lines' not in dict_collect:
                dict_collect['lines'] = []
            dict_collect['lines'].append(line_arry)
    elif geo_dict['type'] == 'LineString':
        line_arry = np.zeros([len(geo_dict['coordinates']), 2])
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
        from osgeo import ogr
        geo = ogr.CreateGeometryFromJson(json.dumps(geo_dict))
        if geo.HasCurveGeometry():
            geo = geo.GetLinearGeometry()
            geo = geo.ExportToJson()
            geo = json.loads(geo)
            _flat_geoms(geo, dict_collect)
        else:
            raise RuntimeError(f"unsupported geometry: {geo_dict}")

def _plot_collection(ax, plot_collect, **style_kwds):
    if len(plot_collect) == 0:
        return None

    try:
        from descartes.patch import PolygonPatch
    except ImportError:
        raise ImportError(
            "The descartes package is required for plotting polygons in geopandas. "
            "You can install it using 'conda install -c conda-forge descartes' "
        )
    try:
        from matplotlib.collections import PatchCollection, LineCollection
    except ImportError:
        raise ImportError(
            "The matplotlib package is required for plotting polygons in geopandas. "
            "You can install it using 'conda install -c conda-forge descartes' "
        )

    if 'polygons' in plot_collect:
        collection = PatchCollection([PolygonPatch(geo) for geo in plot_collect['polygons']])
        ax.add_collection(collection, autolim=True)
    if 'lines' in plot_collect:
        collection = LineCollection(plot_collect['lines'])
        ax.add_collection(collection, autolim=True)
    if 'points' in plot_collect:
        x = [p[0] for p in plot_collect['points']]
        y = [p[1] for p in plot_collect['points']]
        collection = ax.scatter(x, y)
    ax.autoscale_view()
    return None

def _plot_pandas_series(ax, geoms, **style_kwds):
    import pandas.core.series
    import json
    if not isinstance(geoms, pandas.core.series.Series):
        raise TypeError("geoms shuld be type of pandas.core.series.Series")
    if len(geoms) < 1:
        return None
    if isinstance(geoms[0], str):
        pass
    elif isinstance(geoms[0], bytes):
        import arctern
        geoms = arctern.ST_AsGeoJSON(geoms)
    else:
        raise RuntimeError(f"unexpected input type, {type(geoms[0])}")

    plot_collect = dict()
    for geo in geoms:
        geo_dict = json.loads(geo)
        _flat_geoms(geo_dict, plot_collect)

    _plot_collection(ax, plot_collect, **style_kwds)
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
