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

import sys
import pandas as pd
import numpy as np
import geopandas as gpd
import shapely
import arctern
from arctern import GeoSeries


if sys.version_info[0] < 3:
    from StringIO import StringIO
else:
    from io import StringIO

nyc_schema = {
    "VendorID": "string",
    "tpep_pickup_datetime": "string",
    "tpep_dropoff_datetime": "string",
    "passenger_count": "int64",
    "trip_distance": "double",
    "pickup_longitude": "double",
    "pickup_latitude": "double",
    "dropoff_longitude": "double",
    "dropoff_latitude": "double",
    "fare_amount": "double",
    "tip_amount": "double",
    "total_amount": "double",
    "buildingid_pickup": "int64",
    "buildingid_dropoff": "int64",
    "buildingtext_pickup": "string",
    "buildingtext_dropoff": "string",
}

# pylint: disable=too-many-lines,unused-variable,bare-except,broad-except,unidiomatic-typecheck
TESTDATA = StringIO("""
VendorID,tpep_pickup_datetime,tpep_dropoff_datetime,passenger_count,trip_distance,pickup_longitude,pickup_latitude,dropoff_longitude,dropoff_latitude,fare_amount,tip_amount,total_amount,buildingid_pickup,buildingid_dropoff,buildingtext_pickup,buildingtext_dropoff
CMT,2009-04-12 03:16:33 +00:00,2009-04-12 03:20:32 +00:00,1,1.3999999999999999,-73.993003000000002,40.747593999999999,-73.983609000000001,40.760426000000002,5.7999999999999998,0,5.7999999999999998,0,0,,"POLYGON ((-73.9833003295812 40.7590607716671,-73.983284516568 40.7590540993346,-73.9831103296997 40.7589805990397,-73.9832739116023 40.7587564485334,-73.9832848295013 40.7587414900311,-73.983320325137 40.75875646787,-73.983664080401 40.7589015182181,-73.983623439378 40.7589572077704,-73.9835486408108 40.7590597024757,-73.9834895815413 40.7591406286961,-73.9834880812315 40.7591399954262,-73.9833003295812 40.7590607716671))"
VTS,2009-04-14 11:22:00 +00:00,2009-04-14 11:38:00 +00:00,1,2.1400000000000001,-73.959907999999999,40.776353,-73.98348,40.759042000000001,10.1,2,12.1,0,150047,,"POLYGON ((-73.9833003295812 40.7590607716671,-73.983284516568 40.7590540993346,-73.9831103296997 40.7589805990397,-73.9832739116023 40.7587564485334,-73.9832848295013 40.7587414900311,-73.983320325137 40.75875646787,-73.983664080401 40.7589015182181,-73.983623439378 40.7589572077704,-73.9835486408108 40.7590597024757,-73.9834895815413 40.7591406286961,-73.9834880812315 40.7591399954262,-73.9833003295812 40.7590607716671))"
CMT,2009-04-15 09:34:58 +00:00,2009-04-15 09:49:35 +00:00,1,2.7000000000000002,-73.955183000000005,40.773459000000003,-73.985134000000002,40.759250999999999,10.1,1,11.1,0,0,,"POLYGON ((-73.9833003295812 40.7590607716671,-73.983284516568 40.7590540993346,-73.9831103296997 40.7589805990397,-73.9832739116023 40.7587564485334,-73.9832848295013 40.7587414900311,-73.983320325137 40.75875646787,-73.983664080401 40.7589015182181,-73.983623439378 40.7589572077704,-73.9835486408108 40.7590597024757,-73.9834895815413 40.7591406286961,-73.9834880812315 40.7591399954262,-73.9833003295812 40.7590607716671))"
CMT,2009-04-30 18:58:19 +00:00,2009-04-30 19:05:27 +00:00,1,1.3,-73.985232999999994,40.744681999999997,-73.983243999999999,40.758766000000001,5.7000000000000002,0,5.7000000000000002,0,365034,,"POLYGON ((-73.9822052908304 40.7588972120254,-73.9822071211869 40.7588947035016,-73.9822567634792 40.7588266834214,-73.9821224241925 40.7587699956835,-73.9818128940841 40.758639381233,-73.9820162460964 40.758360744719,-73.9818382732697 40.7582856435055,-73.981819409121 40.7582776827681,-73.981899400788 40.7581680769012,-73.9820251917198 40.7582211579493,-73.9822855828536 40.7583310373706,-73.9823738081397 40.7583682660693,-73.9823753913099 40.7583689344862,-73.98232066007 40.7584439282515,-73.9828398129978 40.7586629960315,-73.982820729027 40.7586891456491,-73.9829388887601 40.758739005252,-73.9830080346481 40.7586442571473,-73.9830174698051 40.7586482387701,-73.9832739116023 40.7587564485334,-73.9831103296997 40.7589805990397,-73.9829993050139 40.7589337510552,-73.9829563840912 40.7589925629862,-73.9828327458205 40.7591619782081,-73.9822105696801 40.7588994397891,-73.9822052908304 40.7588972120254))"
CMT,2009-04-26 13:03:04 +00:00,2009-04-26 13:27:54 +00:00,1,8.1999999999999993,-73.997968999999998,40.682816000000003,-73.983288999999999,40.758235999999997,21.699999999999999,0,21.699999999999999,0,0,,"POLYGON ((-73.9833003295812 40.7590607716671,-73.983284516568 40.7590540993346,-73.9831103296997 40.7589805990397,-73.9832739116023 40.7587564485334,-73.9832848295013 40.7587414900311,-73.983320325137 40.75875646787,-73.983664080401 40.7589015182181,-73.983623439378 40.7589572077704,-73.9835486408108 40.7590597024757,-73.9834895815413 40.7591406286961,-73.9834880812315 40.7591399954262,-73.9833003295812 40.7590607716671))"
VTS,2009-04-03 02:56:00 +00:00,2009-04-03 03:11:00 +00:00,5,2.7599999999999998,-73.996458000000004,40.758197000000003,-73.987071999999998,40.759524999999996,10.1,0,10.6,0,342186,,"POLYGON ((-73.9869173687449 40.7597622353379,-73.9868375526983 40.7597283915828,-73.9866791476551 40.7596612245398,-73.9868411600762 40.7594403097268,-73.9868522525914 40.7594251843146,-73.9870256216776 40.7594986968641,-73.9870764477092 40.7595202480534,-73.9873175334296 40.7596224722278,-73.9872933750219 40.7596554136303,-73.9871975549645 40.7597860716577,-73.9871620385817 40.7598345013504,-73.9871444284451 40.7598585131452,-73.9869376442741 40.7597708320555,-73.9869173687449 40.7597622353379))"
VTS,2009-04-02 17:03:00 +00:00,2009-04-02 17:07:00 +00:00,3,0.76000000000000001,-73.988240000000005,40.748959999999997,-73.981067999999993,40.759422999999998,4.5,0,5.5,0,0,,"POLYGON ((-73.9833003295812 40.7590607716671,-73.983284516568 40.7590540993346,-73.9831103296997 40.7589805990397,-73.9832739116023 40.7587564485334,-73.9832848295013 40.7587414900311,-73.983320325137 40.75875646787,-73.983664080401 40.7589015182181,-73.983623439378 40.7589572077704,-73.9835486408108 40.7590597024757,-73.9834895815413 40.7591406286961,-73.9834880812315 40.7591399954262,-73.9833003295812 40.7590607716671))"
VTS,2009-04-23 08:10:00 +00:00,2009-04-23 08:21:00 +00:00,1,1.99,-73.985185000000001,40.735827999999998,-73.981579999999994,40.759551999999999,7.7000000000000002,0,7.7000000000000002,0,0,,"POLYGON ((-73.9833003295812 40.7590607716671,-73.983284516568 40.7590540993346,-73.9831103296997 40.7589805990397,-73.9832739116023 40.7587564485334,-73.9832848295013 40.7587414900311,-73.983320325137 40.75875646787,-73.983664080401 40.7589015182181,-73.983623439378 40.7589572077704,-73.9835486408108 40.7590597024757,-73.9834895815413 40.7591406286961,-73.9834880812315 40.7591399954262,-73.9833003295812 40.7590607716671))"
CMT,2009-04-21 12:18:15 +00:00,2009-04-21 12:29:33 +00:00,1,0.90000000000000002,-73.989726000000005,40.767795,-73.982844,40.759284000000001,7.2999999999999998,0,7.2999999999999998,123894,0,"POLYGON ((-73.989754263774 40.7677468202825,-73.9899519048903 40.7678302792556,-73.989912476786 40.7678842519974,-73.9899105593281 40.7678834422768,-73.9899028933374 40.7678939333729,-73.9897724980032 40.7678388704833,-73.989737963688 40.7678242873584,-73.9897071707312 40.7678112849412,-73.9897080734511 40.7678100513318,-73.9897150393223 40.7678005156204,-73.989754263774 40.7677468202825))","POLYGON ((-73.9833003295812 40.7590607716671,-73.983284516568 40.7590540993346,-73.9831103296997 40.7589805990397,-73.9832739116023 40.7587564485334,-73.9832848295013 40.7587414900311,-73.983320325137 40.75875646787,-73.983664080401 40.7589015182181,-73.983623439378 40.7589572077704,-73.9835486408108 40.7590597024757,-73.9834895815413 40.7591406286961,-73.9834880812315 40.7591399954262,-73.9833003295812 40.7590607716671))"
CMT,2009-04-10 08:54:21 +00:00,2009-04-10 09:07:14 +00:00,1,1.3,-73.992669000000006,40.768326999999999,-73.982506999999998,40.758156999999997,8.0999999999999996,0,8.0999999999999996,0,0,,"POLYGON ((-73.9833003295812 40.7590607716671,-73.983284516568 40.7590540993346,-73.9831103296997 40.7589805990397,-73.9832739116023 40.7587564485334,-73.9832848295013 40.7587414900311,-73.983320325137 40.75875646787,-73.983664080401 40.7589015182181,-73.983623439378 40.7589572077704,-73.9835486408108 40.7590597024757,-73.9834895815413 40.7591406286961,-73.9834880812315 40.7591399954262,-73.9833003295812 40.7590607716671))"
""")

df = pd.read_csv(TESTDATA,
                 dtype=nyc_schema,
                 date_parser=pd.to_datetime,
                 parse_dates=["tpep_pickup_datetime", "tpep_dropoff_datetime"])

geo_dropoff = df['buildingtext_dropoff'].dropna().head(10)
geo_pickup = df['buildingtext_pickup'].dropna().head(10)
VendorID = df['VendorID'].dropna().head(10)
trip_distance = df['trip_distance'].dropna().head(10)

arctern_df = df.copy()
gpd_df = df.copy()
arctern_df['test_GeoSeries'] = arctern.GeoSeries(df['buildingtext_dropoff'])
#gpd_df['test_GeoSeries'] = gpd.GeoSeries(df['buildingtext_dropoff'].apply(shapely.wkt.loads)).apply(
#    lambda x: x.to_wkb())
gpd_df['test_GeoSeries'] = gpd.GeoSeries(df['buildingtext_dropoff'].apply(shapely.wkt.loads)).apply(
    lambda x: x.wkb)
pd.testing.assert_frame_equal(arctern_df, gpd_df, check_dtype=False)


def trans2wkb(_df, key, index=range(0, 0)):
    if isinstance(index, range):
        index = range(0, _df[key].count())
    import pygeos
    s_arr = []
    if not isinstance(_df, pd.DataFrame):
        return None
    try:
        s = _df[key]
        for i in range(0, s.count()):
            s_arr.append(pygeos.to_wkb(pygeos.Geometry(s[i])))
        _df[key] = pd.Series(s_arr, index=index)
    except:
        return None
    return _df


def trans2wkb4series(s, index=range(0, 0)):
    if isinstance(index, range):
        index = range(0, s.size)
    import pygeos
    s_arr = []
    if not isinstance(s, pd.Series):
        return None
    try:
        for i in range(0, s.size):
            if not s[i]:
                s_arr.append(None)
            else:
                s_arr.append(pygeos.to_wkb(pygeos.Geometry(s[i])))
        s = pd.Series(s_arr, index=index)
    except:
        return None
    return s


def test_DataFrame():
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html
    geo_s = GeoSeries(geo_dropoff.to_list())
    pd_s = pd.Series(geo_dropoff.to_list())
    pd_s_wkb = trans2wkb4series(pd_s)
    pd.testing.assert_series_equal(geo_s, pd_s_wkb, check_dtype=False)

    geo_df = pd.DataFrame({'s': geo_s})
    pd_df_wkb = pd.DataFrame({'s': pd_s_wkb})
    pd.testing.assert_frame_equal(geo_df, pd_df_wkb, check_dtype=False)


def test_index():
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.index.html
    geo_s = GeoSeries(geo_dropoff.to_list())
    pd_s = pd.Series(geo_dropoff.to_list())
    pd_s_wkb = trans2wkb4series(pd_s)
    pd.testing.assert_series_equal(geo_s, pd_s_wkb, check_dtype=False)

    geo_df = pd.DataFrame({'s': geo_s})
    pd_df_wkb = pd.DataFrame({'s': pd_s_wkb})
    pd.testing.assert_frame_equal(geo_df, pd_df_wkb, check_dtype=False)

    geo_res = geo_df.index
    pd_res = pd_df_wkb.index

    assert (geo_res == pd_res).all()


def test_columns():
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.columns.html
    geo_s = GeoSeries(geo_dropoff.to_list())
    pd_s = pd.Series(geo_dropoff.to_list())
    pd_s_wkb = trans2wkb4series(pd_s)
    pd.testing.assert_series_equal(geo_s, pd_s_wkb, check_dtype=False)

    geo_df = pd.DataFrame({'s': geo_s})
    pd_df_wkb = pd.DataFrame({'s': pd_s_wkb})
    pd.testing.assert_frame_equal(geo_df, pd_df_wkb, check_dtype=False)

    geo_res = geo_df.columns
    pd_res = pd_df_wkb.columns

    assert (geo_res == pd_res).all()


def test_dtypes():
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.dtypes.html
    geo_s = GeoSeries(geo_dropoff.to_list())
    pd_s = pd.Series(geo_dropoff.to_list())
    pd_s_wkb = trans2wkb4series(pd_s)
    pd.testing.assert_series_equal(geo_s, pd_s_wkb, check_dtype=False)

    geo_df = pd.DataFrame({'s': geo_s})
    pd_df_wkb = pd.DataFrame({'s': pd_s_wkb})
    pd.testing.assert_frame_equal(geo_df, pd_df_wkb, check_dtype=False)

    geo_res = geo_df.dtypes
    pd_res = pd_df_wkb.dtypes
    assert True  # GeoSeries' default output type is GeoDtype, which is different from pandas, which is correct


def test_select_dtypes():
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.select_dtypes.html
    geo_s = GeoSeries(geo_dropoff.to_list())
    pd_s = pd.Series(geo_dropoff.to_list())
    pd_s_wkb = trans2wkb4series(pd_s)
    pd.testing.assert_series_equal(geo_s, pd_s_wkb, check_dtype=False)

    geo_df = pd.DataFrame({'s': geo_s})
    pd_df_wkb = pd.DataFrame({'s': pd_s_wkb})
    pd.testing.assert_frame_equal(geo_df, pd_df_wkb, check_dtype=False)

    geo_res = geo_df.select_dtypes(exclude=['int'])
    pd_res = pd_df_wkb.select_dtypes(exclude=['int'])
    pd.testing.assert_frame_equal(geo_res, pd_res, check_dtype=False)


def test_values():
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.values.html
    geo_s = GeoSeries(geo_dropoff.to_list())
    pd_s = pd.Series(geo_dropoff.to_list())
    pd_s_wkb = trans2wkb4series(pd_s)
    pd.testing.assert_series_equal(geo_s, pd_s_wkb, check_dtype=False)

    geo_df = pd.DataFrame({'s': geo_s})
    pd_df_wkb = pd.DataFrame({'s': pd_s_wkb})
    pd.testing.assert_frame_equal(geo_df, pd_df_wkb, check_dtype=False)

    geo_res = geo_df.values
    pd_res = pd_df_wkb.values

    assert (geo_res == pd_res).all()


def test_axes():
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.axes.html
    geo_s = GeoSeries(geo_dropoff.to_list())
    pd_s = pd.Series(geo_dropoff.to_list())
    pd_s_wkb = trans2wkb4series(pd_s)
    pd.testing.assert_series_equal(geo_s, pd_s_wkb, check_dtype=False)

    geo_df = pd.DataFrame({'s': geo_s})
    pd_df_wkb = pd.DataFrame({'s': pd_s_wkb})
    pd.testing.assert_frame_equal(geo_df, pd_df_wkb, check_dtype=False)

    geo_res = geo_df.axes
    pd_res = pd_df_wkb.axes

    assert (geo_res[0] == pd_res[0]).all()
    assert geo_res[1] == pd_res[1]


def test_ndim():
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.ndim.html
    geo_s = GeoSeries(geo_dropoff.to_list())
    pd_s = pd.Series(geo_dropoff.to_list())
    pd_s_wkb = trans2wkb4series(pd_s)
    pd.testing.assert_series_equal(geo_s, pd_s_wkb, check_dtype=False)

    geo_df = pd.DataFrame({'s': geo_s})
    pd_df_wkb = pd.DataFrame({'s': pd_s_wkb})
    pd.testing.assert_frame_equal(geo_df, pd_df_wkb, check_dtype=False)

    geo_res = geo_df.ndim
    pd_res = pd_df_wkb.ndim

    assert geo_res == pd_res


def test_size():
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.size.html
    geo_s = GeoSeries(geo_dropoff.to_list())
    pd_s = pd.Series(geo_dropoff.to_list())
    pd_s_wkb = trans2wkb4series(pd_s)
    pd.testing.assert_series_equal(geo_s, pd_s_wkb, check_dtype=False)

    geo_df = pd.DataFrame({'s': geo_s})
    pd_df_wkb = pd.DataFrame({'s': pd_s_wkb})
    pd.testing.assert_frame_equal(geo_df, pd_df_wkb, check_dtype=False)

    geo_res = geo_df.size
    pd_res = pd_df_wkb.size

    assert geo_res == pd_res


def test_shape():
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.shape.html
    geo_s = GeoSeries(geo_dropoff.to_list())
    pd_s = pd.Series(geo_dropoff.to_list())
    pd_s_wkb = trans2wkb4series(pd_s)
    pd.testing.assert_series_equal(geo_s, pd_s_wkb, check_dtype=False)

    geo_df = pd.DataFrame({'s': geo_s})
    pd_df_wkb = pd.DataFrame({'s': pd_s_wkb})
    pd.testing.assert_frame_equal(geo_df, pd_df_wkb, check_dtype=False)

    geo_res = geo_df.shape
    pd_res = pd_df_wkb.shape

    assert geo_res == pd_res


def test_memory_usage():
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.memory_usage.html
    geo_s = GeoSeries(geo_dropoff.to_list())
    pd_s = pd.Series(geo_dropoff.to_list())
    pd_s_wkb = trans2wkb4series(pd_s)
    pd.testing.assert_series_equal(geo_s, pd_s_wkb, check_dtype=False)

    geo_df = pd.DataFrame({'s': geo_s})
    pd_df_wkb = pd.DataFrame({'s': pd_s_wkb})
    pd.testing.assert_frame_equal(geo_df, pd_df_wkb, check_dtype=False)

    geo_res = geo_df.memory_usage()
    pd_res = pd_df_wkb.memory_usage()

    assert (geo_res == pd_res).all()


def test_empty():
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.empty.html
    geo_s = GeoSeries(geo_dropoff.to_list())
    pd_s = pd.Series(geo_dropoff.to_list())
    pd_s_wkb = trans2wkb4series(pd_s)
    pd.testing.assert_series_equal(geo_s, pd_s_wkb, check_dtype=False)

    geo_df = pd.DataFrame({'s': geo_s})
    pd_df_wkb = pd.DataFrame({'s': pd_s_wkb})
    pd.testing.assert_frame_equal(geo_df, pd_df_wkb, check_dtype=False)

    geo_res = geo_df.empty
    pd_res = pd_df_wkb.empty

    assert geo_res == pd_res


def test_astype():
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.astype.html
    geo_s = GeoSeries(geo_dropoff.to_list())
    pd_s = pd.Series(geo_dropoff.to_list())
    pd_s_wkb = trans2wkb4series(pd_s)
    pd.testing.assert_series_equal(geo_s, pd_s_wkb, check_dtype=False)

    geo_df = pd.DataFrame({'s': geo_s})
    pd_df_wkb = pd.DataFrame({'s': pd_s_wkb})
    pd.testing.assert_frame_equal(geo_df, pd_df_wkb, check_dtype=False)

    geo_res = geo_df.astype('object').dtypes
    pd_res = pd_df_wkb.astype('object').dtypes

    assert (geo_res == pd_res).all()


def test_convert_dtypes():
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.convert_dtypes.html
    geo_s = GeoSeries(geo_dropoff.to_list())
    pd_s = pd.Series(geo_dropoff.to_list())
    pd_s_wkb = trans2wkb4series(pd_s)
    pd.testing.assert_series_equal(geo_s, pd_s_wkb, check_dtype=False)

    geo_df = pd.DataFrame({'s': geo_s})
    pd_df_wkb = pd.DataFrame({'s': pd_s_wkb})
    pd.testing.assert_frame_equal(geo_df, pd_df_wkb, check_dtype=False)

    geo_res = geo_df.convert_dtypes()
    pd_res = pd_df_wkb.convert_dtypes()

    pd.testing.assert_frame_equal(geo_res, pd_res, check_dtype=False)


def test_infer_objects():
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.infer_objects.html
    geo_s = GeoSeries(geo_dropoff.to_list())
    pd_s = pd.Series(geo_dropoff.to_list())
    pd_s_wkb = trans2wkb4series(pd_s)
    pd.testing.assert_series_equal(geo_s, pd_s_wkb, check_dtype=False)

    geo_df = pd.DataFrame({'s': geo_s})
    pd_df_wkb = pd.DataFrame({'s': pd_s_wkb})
    pd.testing.assert_frame_equal(geo_df, pd_df_wkb, check_dtype=False)

    geo_res = geo_df.infer_objects().dtypes
    pd_res = pd_df_wkb.infer_objects().dtypes

    assert True  # GeoSeries' default output type is GeoDtype, which is different from pandas, which is correct


def test_copy():
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.copy.html
    geo_s = GeoSeries(geo_dropoff.to_list())
    pd_s = pd.Series(geo_dropoff.to_list())
    pd_s_wkb = trans2wkb4series(pd_s)
    pd.testing.assert_series_equal(geo_s, pd_s_wkb, check_dtype=False)

    geo_df = pd.DataFrame({'s': geo_s})
    pd_df_wkb = pd.DataFrame({'s': pd_s_wkb})
    pd.testing.assert_frame_equal(geo_df, pd_df_wkb, check_dtype=False)

    geo_res = geo_df.copy()
    pd_res = pd_df_wkb.copy()

    pd.testing.assert_frame_equal(geo_res, pd_res, check_dtype=False)


def test_isna():
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.isna.html
    geo_s = GeoSeries(geo_dropoff.to_list())
    pd_s = pd.Series(geo_dropoff.to_list())
    pd_s_wkb = trans2wkb4series(pd_s)
    pd.testing.assert_series_equal(geo_s, pd_s_wkb, check_dtype=False)

    geo_df = pd.DataFrame({'s': geo_s})
    pd_df_wkb = pd.DataFrame({'s': pd_s_wkb})
    pd.testing.assert_frame_equal(geo_df, pd_df_wkb, check_dtype=False)

    geo_res = geo_df.isna()
    pd_res = pd_df_wkb.isna()

    pd.testing.assert_frame_equal(geo_res, pd_res, check_dtype=True)


def test_notna():
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.notna.html
    geo_s = GeoSeries(geo_dropoff.to_list())
    pd_s = pd.Series(geo_dropoff.to_list())
    pd_s_wkb = trans2wkb4series(pd_s)
    pd.testing.assert_series_equal(geo_s, pd_s_wkb, check_dtype=False)

    geo_df = pd.DataFrame({'s': geo_s})
    pd_df_wkb = pd.DataFrame({'s': pd_s_wkb})
    pd.testing.assert_frame_equal(geo_df, pd_df_wkb, check_dtype=False)

    geo_res = geo_df.notna()
    pd_res = pd_df_wkb.notna()

    pd.testing.assert_frame_equal(geo_res, pd_res, check_dtype=True)


def test_bool():
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.bool.html
    geo_s = GeoSeries(geo_dropoff.to_list())
    pd_s = pd.Series(geo_dropoff.to_list())
    pd_s_wkb = trans2wkb4series(pd_s)
    pd.testing.assert_series_equal(geo_s, pd_s_wkb, check_dtype=False)

    geo_df = pd.DataFrame({'s': geo_s})
    pd_df_wkb = pd.DataFrame({'s': pd_s_wkb})
    pd.testing.assert_frame_equal(geo_df, pd_df_wkb, check_dtype=False)

    geo_res = geo_df.all().bool()
    pd_res = pd_df_wkb.all().bool()

    assert geo_res == pd_res


def test_head():
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.head.html
    geo_s = GeoSeries(geo_dropoff.to_list())
    pd_s = pd.Series(geo_dropoff.to_list())
    pd_s_wkb = trans2wkb4series(pd_s)
    pd.testing.assert_series_equal(geo_s, pd_s_wkb, check_dtype=False)

    geo_df = pd.DataFrame({'s': geo_s})
    pd_df_wkb = pd.DataFrame({'s': pd_s_wkb})
    pd.testing.assert_frame_equal(geo_df, pd_df_wkb, check_dtype=False)

    geo_res = geo_df.head()
    pd_res = pd_df_wkb.head()

    pd.testing.assert_frame_equal(geo_res, pd_res, check_dtype=False)


def test_at():
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.at.html
    geo_s = GeoSeries(geo_dropoff.to_list())
    pd_s = pd.Series(geo_dropoff.to_list())
    pd_s_wkb = trans2wkb4series(pd_s)
    pd.testing.assert_series_equal(geo_s, pd_s_wkb, check_dtype=False)

    geo_df = pd.DataFrame({'s': geo_s})
    pd_df_wkb = pd.DataFrame({'s': pd_s_wkb})
    pd.testing.assert_frame_equal(geo_df, pd_df_wkb, check_dtype=False)

    geo_res = geo_df.at.obj
    pd_res = pd_df_wkb.at.obj

    assert (geo_res == pd_res).all().all()


def test_iat():
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.iat.html
    geo_s = GeoSeries(geo_dropoff.to_list())
    pd_s = pd.Series(geo_dropoff.to_list())
    pd_s_wkb = trans2wkb4series(pd_s)
    pd.testing.assert_series_equal(geo_s, pd_s_wkb, check_dtype=False)

    geo_df = pd.DataFrame({'s': geo_s})
    pd_df_wkb = pd.DataFrame({'s': pd_s_wkb})
    pd.testing.assert_frame_equal(geo_df, pd_df_wkb, check_dtype=False)

    geo_res = geo_df.iat.obj
    pd_res = pd_df_wkb.iat.obj

    pd.testing.assert_frame_equal(geo_res, pd_res, check_dtype=False)


def test_loc():
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.loc.html
    geo_s = GeoSeries(geo_dropoff.to_list())
    pd_s = pd.Series(geo_dropoff.to_list())
    pd_s_wkb = trans2wkb4series(pd_s)
    pd.testing.assert_series_equal(geo_s, pd_s_wkb, check_dtype=False)

    geo_df = pd.DataFrame({'s': geo_s})
    pd_df_wkb = pd.DataFrame({'s': pd_s_wkb})
    pd.testing.assert_frame_equal(geo_df, pd_df_wkb, check_dtype=False)

    geo_res = geo_df.loc.obj
    pd_res = pd_df_wkb.loc.obj

    pd.testing.assert_frame_equal(geo_res, pd_res, check_dtype=False)


def test_iloc():
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.iloc.html
    geo_s = GeoSeries(geo_dropoff.to_list())
    pd_s = pd.Series(geo_dropoff.to_list())
    pd_s_wkb = trans2wkb4series(pd_s)
    pd.testing.assert_series_equal(geo_s, pd_s_wkb, check_dtype=False)

    geo_df = pd.DataFrame({'s': geo_s})
    pd_df_wkb = pd.DataFrame({'s': pd_s_wkb})
    pd.testing.assert_frame_equal(geo_df, pd_df_wkb, check_dtype=False)

    geo_res = geo_df.iloc.obj
    pd_res = pd_df_wkb.iloc.obj

    pd.testing.assert_frame_equal(geo_res, pd_res, check_dtype=False)


def test_insert():
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.insert.html
    geo_s = GeoSeries(geo_dropoff.to_list())
    pd_s = pd.Series(geo_dropoff.to_list())
    pd_s_wkb = trans2wkb4series(pd_s)
    pd.testing.assert_series_equal(geo_s, pd_s_wkb, check_dtype=False)

    geo_df = pd.DataFrame({'s': geo_s})
    pd_df_wkb = pd.DataFrame({'s': pd_s_wkb})
    pd.testing.assert_frame_equal(geo_df, pd_df_wkb, check_dtype=False)

    geo_df.insert(0, 'q', 'POINT (1 2)')
    pd_df_wkb.insert(0, 'q', 'POINT (1 2)')
    pd.testing.assert_frame_equal(geo_df, pd_df_wkb, check_dtype=False)


def test_iter():
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.__iter__.html
    geo_s = GeoSeries(geo_dropoff.to_list())
    pd_s = pd.Series(geo_dropoff.to_list())
    pd_s_wkb = trans2wkb4series(pd_s)
    pd.testing.assert_series_equal(geo_s, pd_s_wkb, check_dtype=False)

    geo_df = pd.DataFrame({'s': geo_s})
    pd_df_wkb = pd.DataFrame({'s': pd_s_wkb})
    pd.testing.assert_frame_equal(geo_df, pd_df_wkb, check_dtype=False)

    geo_res = geo_df.__iter__().__next__()
    pd_res = pd_df_wkb.__iter__().__next__()
    assert geo_res == pd_res


def test_items():
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.items.html
    geo_s = GeoSeries(geo_dropoff.to_list())
    pd_s = pd.Series(geo_dropoff.to_list())
    pd_s_wkb = trans2wkb4series(pd_s)
    pd.testing.assert_series_equal(geo_s, pd_s_wkb, check_dtype=False)

    geo_df = pd.DataFrame({'s': geo_s})
    pd_df_wkb = pd.DataFrame({'s': pd_s_wkb})
    pd.testing.assert_frame_equal(geo_df, pd_df_wkb, check_dtype=False)

    geo_res = []
    pd_res = []
    for i in geo_df.items():
        geo_res.append(i)
    for j in pd_df_wkb.items():
        pd_res.append(j)

    pd.testing.assert_series_equal(geo_res[0][1], pd_res[0][1], check_dtype=False)


def test_iteritems():
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.iteritems.html
    geo_s = GeoSeries(geo_dropoff.to_list())
    pd_s = pd.Series(geo_dropoff.to_list())
    pd_s_wkb = trans2wkb4series(pd_s)
    pd.testing.assert_series_equal(geo_s, pd_s_wkb, check_dtype=False)

    geo_df = pd.DataFrame({'s': geo_s})
    pd_df_wkb = pd.DataFrame({'s': pd_s_wkb})
    pd.testing.assert_frame_equal(geo_df, pd_df_wkb, check_dtype=False)

    geo_res = []
    pd_res = []
    for i in geo_df.iteritems():
        geo_res.append(i)
    for j in pd_df_wkb.iteritems():
        pd_res.append(j)

    pd.testing.assert_series_equal(geo_res[0][1], pd_res[0][1], check_dtype=False)


def test_keys():
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.keys.html
    geo_s = GeoSeries(geo_dropoff.to_list())
    pd_s = pd.Series(geo_dropoff.to_list())
    pd_s_wkb = trans2wkb4series(pd_s)
    pd.testing.assert_series_equal(geo_s, pd_s_wkb, check_dtype=False)

    geo_df = pd.DataFrame({'s': geo_s})
    pd_df_wkb = pd.DataFrame({'s': pd_s_wkb})
    pd.testing.assert_frame_equal(geo_df, pd_df_wkb, check_dtype=False)

    geo_res = geo_df.keys()
    pd_res = pd_df_wkb.keys()

    assert geo_res == pd_res


def test_iterrows():
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.iterrows.html
    geo_s = GeoSeries(geo_dropoff.to_list())
    pd_s = pd.Series(geo_dropoff.to_list())
    pd_s_wkb = trans2wkb4series(pd_s)
    pd.testing.assert_series_equal(geo_s, pd_s_wkb, check_dtype=False)

    geo_df = pd.DataFrame({'s': geo_s})
    pd_df_wkb = pd.DataFrame({'s': pd_s_wkb})
    pd.testing.assert_frame_equal(geo_df, pd_df_wkb, check_dtype=False)

    geo_res = []
    pd_res = []
    for i in geo_df.iterrows():
        geo_res.append(i)
    for j in pd_df_wkb.iterrows():
        pd_res.append(j)

    pd.testing.assert_series_equal(geo_res[0][1], pd_res[0][1], check_dtype=False)


def test_itertuples():
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.itertuples.html
    geo_s = GeoSeries(geo_dropoff.to_list())
    pd_s = pd.Series(geo_dropoff.to_list())
    pd_s_wkb = trans2wkb4series(pd_s)
    pd.testing.assert_series_equal(geo_s, pd_s_wkb, check_dtype=False)

    geo_df = pd.DataFrame({'s': geo_s})
    pd_df_wkb = pd.DataFrame({'s': pd_s_wkb})
    pd.testing.assert_frame_equal(geo_df, pd_df_wkb, check_dtype=False)

    geo_res = []
    pd_res = []
    for i in geo_df.itertuples():
        geo_res.append(i)

    for j in pd_df_wkb.itertuples():
        pd_res.append(j)

    assert geo_res == pd_res


def test_lookup():
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.lookup.html
    geo_s = GeoSeries(geo_dropoff.to_list())
    pd_s = pd.Series(geo_dropoff.to_list())
    pd_s_wkb = trans2wkb4series(pd_s)
    pd.testing.assert_series_equal(geo_s, pd_s_wkb, check_dtype=False)

    geo_df = pd.DataFrame({'s': geo_s})
    pd_df_wkb = pd.DataFrame({'s': pd_s_wkb})
    pd.testing.assert_frame_equal(geo_df, pd_df_wkb, check_dtype=False)

    pd_res = pd_df_wkb.lookup(pd_df_wkb.index, ['s' for i in range(0, 10)])
    geo_res = geo_df.lookup(geo_df.index, ['s' for i in range(0, 10)])
    assert (geo_res == pd_res).all()


def test_pop():
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.pop.html
    geo_s = GeoSeries(geo_dropoff.to_list())
    pd_s = pd.Series(geo_dropoff.to_list())
    pd_s_wkb = trans2wkb4series(pd_s)
    pd.testing.assert_series_equal(geo_s, pd_s_wkb, check_dtype=False)

    geo_df = pd.DataFrame({'s': geo_s})
    pd_df_wkb = pd.DataFrame({'s': pd_s_wkb})
    pd.testing.assert_frame_equal(geo_df, pd_df_wkb, check_dtype=False)

    geo_res = geo_df.pop('s')
    pd_res = pd_df_wkb.pop('s')

    pd.testing.assert_series_equal(geo_res, pd_res, check_dtype=False)


def test_tail():
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.tail.html
    geo_s = GeoSeries(geo_dropoff.to_list())
    pd_s = pd.Series(geo_dropoff.to_list())
    pd_s_wkb = trans2wkb4series(pd_s)
    pd.testing.assert_series_equal(geo_s, pd_s_wkb, check_dtype=False)

    geo_df = pd.DataFrame({'s': geo_s})
    pd_df_wkb = pd.DataFrame({'s': pd_s_wkb})
    pd.testing.assert_frame_equal(geo_df, pd_df_wkb, check_dtype=False)

    geo_res = geo_df.tail(3)
    pd_res = pd_df_wkb.tail(3)

    pd.testing.assert_frame_equal(geo_res, pd_res, check_dtype=False)

    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.xs.html
    # GeoSeries not supported


def test_get():
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.get.html
    geo_s = GeoSeries(geo_dropoff.to_list())
    pd_s = pd.Series(geo_dropoff.to_list())
    pd_s_wkb = trans2wkb4series(pd_s)
    pd.testing.assert_series_equal(geo_s, pd_s_wkb, check_dtype=False)

    geo_df = pd.DataFrame({'s': geo_s})
    pd_df_wkb = pd.DataFrame({'s': pd_s_wkb})
    pd.testing.assert_frame_equal(geo_df, pd_df_wkb, check_dtype=False)

    geo_res = geo_df.get('s')
    pd_res = pd_df_wkb.get('s')

    pd.testing.assert_series_equal(geo_res, pd_res, check_dtype=False)


def test_isin():
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.isin.html
    geo_s = GeoSeries(geo_dropoff.to_list())
    pd_s = pd.Series(geo_dropoff.to_list())
    pd_s_wkb = trans2wkb4series(pd_s)
    pd.testing.assert_series_equal(geo_s, pd_s_wkb, check_dtype=False)

    geo_df = pd.DataFrame({'s': geo_s})
    pd_df_wkb = pd.DataFrame({'s': pd_s_wkb})
    pd.testing.assert_frame_equal(geo_df, pd_df_wkb, check_dtype=False)

    geo_res = geo_df.isin([0, 2])
    pd_res = pd_df_wkb.isin([0, 2])

    pd.testing.assert_frame_equal(geo_res, pd_res, check_dtype=False)


def test_where():
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.where.html
    geo_s = GeoSeries(geo_dropoff.to_list())
    pd_s = pd.Series(geo_dropoff.to_list())
    pd_s_wkb = trans2wkb4series(pd_s)
    pd.testing.assert_series_equal(geo_s, pd_s_wkb, check_dtype=False)

    geo_df = pd.DataFrame({'s': geo_s})
    pd_df_wkb = pd.DataFrame({'s': pd_s_wkb})
    pd.testing.assert_frame_equal(geo_df, pd_df_wkb, check_dtype=False)

    f = geo_df == 0
    geo_res = geo_df.where(f)
    pd_res = pd_df_wkb.where(f)

    pd.testing.assert_frame_equal(geo_res, pd_res, check_dtype=False)


def test_mask():
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.mask.html
    geo_s = GeoSeries(geo_dropoff.to_list())
    pd_s = pd.Series(geo_dropoff.to_list())
    pd_s_wkb = trans2wkb4series(pd_s)
    pd.testing.assert_series_equal(geo_s, pd_s_wkb, check_dtype=False)

    geo_df = pd.DataFrame({'s': geo_s})
    pd_df_wkb = pd.DataFrame({'s': pd_s_wkb})
    pd.testing.assert_frame_equal(geo_df, pd_df_wkb, check_dtype=False)

    f = geo_df == 0
    geo_res = geo_df.mask(f)
    pd_res = pd_df_wkb.mask(f)

    pd.testing.assert_frame_equal(geo_res, pd_res, check_dtype=False)


def test_add():
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.add.html
    geo_s = GeoSeries(geo_dropoff.to_list())
    pd_s = pd.Series(geo_dropoff.to_list())
    pd_s_wkb = trans2wkb4series(pd_s)
    pd.testing.assert_series_equal(geo_s, pd_s_wkb, check_dtype=False)

    geo_df = pd.DataFrame({'s': geo_s})
    pd_df_wkb = pd.DataFrame({'s': pd_s_wkb})
    pd.testing.assert_frame_equal(geo_df, pd_df_wkb, check_dtype=False)

    f = geo_df == 0
    geo_res = geo_df.add(geo_df)
    pd_res = pd_df_wkb.add(pd_df_wkb)

    pd.testing.assert_frame_equal(geo_res, pd_res, check_dtype=False)


def test_rmod():
    try:
        arctern_rmod_df = arctern_df.rmod(df)
    except Exception as e1:
        arctern_rmod_exception = e1
    try:
        gpd_rmod_df = gpd_df.rmod(df)
    except Exception as e2:
        gpd_rmod_exception = e2

    assert type(gpd_rmod_exception) == type(arctern_rmod_exception)


def test_rpow():
    try:
        arctern_rmod_df = arctern_df.rpow(df)
    except Exception as e1:
        arctern_rpow_exception = e1

    try:
        gpd_rmod_df = gpd_df.rpow(df)
    except Exception as e2:
        gpd_rpow_exception = e2

    assert type(gpd_rpow_exception) == type(arctern_rpow_exception)


def test_ne():
    arctern_ne_df = arctern_df.ne(df)
    gpd_ne_df = gpd_df.ne(df)

    assert arctern_ne_df.equals(gpd_ne_df)


def test_eq():
    arctern_eq_df = arctern_df.eq(df)
    gpd_eq_df = gpd_df.eq(df)

    assert arctern_eq_df.equals(gpd_eq_df)


def test_combine():
    take_any = lambda p1, p2: p1 if (p1 == p2).all() else p2
    arctern_combine_df = df.combine(arctern_df, take_any)
    gpd_combine_df = df.combine(gpd_df, take_any)

    pd.testing.assert_frame_equal(arctern_combine_df, gpd_combine_df, check_dtype=False)


def test_combine_first():
    arctern_combine_first_df = arctern_df.combine_first(df)
    gpd_combine_first_df = gpd_df.combine_first(df)

    pd.testing.assert_frame_equal(arctern_combine_first_df, gpd_combine_first_df, check_dtype=False)


def test_apply():
    arctern_apply_series = arctern_df.apply("sum")
    gpd_apply_series = gpd_df.apply("sum")

    pd.testing.assert_series_equal(gpd_apply_series, arctern_apply_series)


def test_applymap():
    arctern_applymap_df = arctern_df.applymap(lambda x: len(str(x)))
    gpd_applymap_df = gpd_df.applymap(lambda x: len(str(x)))

    pd.testing.assert_frame_equal(arctern_applymap_df, gpd_applymap_df)


def test_pipe():
    g_func = lambda x: x
    f_func = lambda x: len(str(x))
    arctern_pipe_obj = arctern_df.pipe(g_func).pipe(f_func)
    gpd_pipe_obj = gpd_df.pipe(g_func).pipe(f_func)

    assert arctern_pipe_obj == gpd_pipe_obj


def test_agg():
    arctern_agg_series = arctern_df.agg('sum')
    gpd_agg_series = gpd_df.agg('sum')

    pd.testing.assert_series_equal(arctern_agg_series, gpd_agg_series)


def test_aggregate():
    arctern_aggregate_series = arctern_df.aggregate('sum', axis=0)
    gpd_aggregate_series = gpd_df.aggregate('sum', axis=0)

    pd.testing.assert_series_equal(arctern_aggregate_series, gpd_aggregate_series)


def transform_func(x):
    return x


def test_transform():
    arctern_transform_df = arctern_df.transform(transform_func)
    gpd_transform_df = gpd_df.transform(transform_func)

    pd.testing.assert_frame_equal(arctern_transform_df, gpd_transform_df, check_dtype=False)


def test_groupby():
    arctern_groupby_df = arctern_df.groupby('test_GeoSeries').mean()
    gpd_groupby_df = gpd_df.groupby('test_GeoSeries').mean()

    pd.testing.assert_frame_equal(arctern_groupby_df, gpd_groupby_df)


def test_abs():
    try:
        arctern_abs_df = arctern.GeoSeries(df['buildingtext_dropoff']).abs()
    except Exception as e1:
        arctern_abs_exception = e1
    try:
        gpd_abs_df = (
            #gpd.GeoSeries(df['buildingtext_dropoff'].apply(shapely.wkt.loads)).apply(lambda x: x.to_wkb())).abs()
            gpd.GeoSeries(df['buildingtext_dropoff'].apply(shapely.wkt.loads)).apply(lambda x: x.wkb)).abs()
    except Exception as e2:
        gpd_abs_exception = e2

    assert type(arctern_abs_exception) == type(gpd_abs_exception)


def test_all():
    arctern_all_series = arctern_df.all()
    gpd_all_series = gpd_df.all()

    pd.testing.assert_series_equal(arctern_all_series, gpd_all_series)


def test_any():
    arctern_any_series = arctern_df.any()
    gpd_any_series = gpd_df.any()

    pd.testing.assert_series_equal(arctern_any_series, gpd_any_series)


def test_clip():
    try:
        arctern_clip_df = arctern.GeoSeries(df['buildingtext_dropoff']).clip(-4, 6)
    except Exception as e1:
        arctern_clip_exception = e1
    try:
        gpd_clip_df = (
            #gpd.GeoSeries(df['buildingtext_dropoff'].apply(shapely.wkt.loads)).apply(lambda x: x.to_wkb())).clip(-4, 6)
            gpd.GeoSeries(df['buildingtext_dropoff'].apply(shapely.wkt.loads)).apply(lambda x: x.wkb)).clip(-4, 6)
    except Exception as e2:
        gpd_clip_exception = e2

    assert type(arctern_clip_exception) == type(gpd_clip_exception)


def test_corr():
    arctern_corr_df = arctern_df.corr()
    gpd_corr_df = gpd_df.corr()

    pd.testing.assert_frame_equal(arctern_corr_df, gpd_corr_df)


def test_corrwith():
    arctern_corrwith_series = arctern_df.corrwith(df)
    gpd_corrwith_series = gpd_df.corrwith(df)

    pd.testing.assert_series_equal(arctern_corrwith_series, gpd_corrwith_series)


def test_count():
    arctern_count_series = arctern_df.count()
    gpd_count_series = gpd_df.count()

    pd.testing.assert_series_equal(arctern_count_series, gpd_count_series)


def test_cov():
    arctern_cov_df = arctern_df.cov()
    gpd_cov_df = gpd_df.cov()

    pd.testing.assert_frame_equal(arctern_cov_df, gpd_cov_df)


def test_cummax():
    try:
        arctern_cummax_df = arctern_df.cummax()
    except Exception as e1:
        arctern_cummax_exception = e1
    try:
        gpd_cummax_df = gpd_df.cummax()
    except Exception as e2:
        gpd_cummax_exception = e2

    assert type(arctern_cummax_exception) == type(gpd_cummax_exception)


def test_cummin():
    try:
        arctern_cummin_df = arctern_df.cummin()
    except Exception as e1:
        arctern_cummin_exception = e1
    try:
        gpd_cummin_df = gpd_df.cummin()
    except Exception as e2:
        gpd_cummin_exception = e2

    assert type(arctern_cummin_exception) == type(gpd_cummin_exception)


def test_cumprod():
    try:
        arctern_cumprod_df = arctern_df.cumprod()
    except Exception as e1:
        arctern_cumprod_exception = e1
    try:
        gpd_cumprod_df = gpd_df.cumprod()
    except Exception as e2:
        gpd_cumprod_exception = e2

    assert type(arctern_cumprod_exception) == type(gpd_cumprod_exception)


def test_cumsum():
    try:
        arctern_cumsum_df = arctern_df.cumsum()
    except Exception as e1:
        arctern_cumsum_exception = e1
    try:
        gpd_cumsum_df = gpd_df.cumsum()
    except Exception as e2:
        gpd_cumsum_exception = e2

    assert type(arctern_cumsum_exception) == type(gpd_cumsum_exception)


def test_describe():
    arctern_describe_df = arctern_df.describe()
    gpd_describe_df = gpd_df.describe()

    pd.testing.assert_frame_equal(arctern_describe_df, gpd_describe_df)


def test_diff():
    try:
        arctern_diff_df = arctern_df.diff()
    except Exception as e1:
        arctern_diff_exception = e1

    try:
        gpd_diff_df = gpd_df.diff()
    except Exception as e2:
        gpd_diff_exception = e2

    assert type(arctern_diff_exception) == type(gpd_diff_exception)


def test_eval():
    arctern_eval_df = arctern_df.eval('geo_copy = test_GeoSeries')
    gpd_eval_df = gpd_df.eval('geo_copy = test_GeoSeries')

    pd.testing.assert_frame_equal(arctern_eval_df, gpd_eval_df, check_dtype=False)


def test_kurt():
    arctern_kurt_series = arctern_df.kurt()
    gpd_kurt_series = gpd_df.kurt()

    pd.testing.assert_series_equal(arctern_kurt_series, gpd_kurt_series)


def test_kurtosis():
    arctern_kurtosis_series = arctern_df.kurtosis()
    gpd_kurtosis_series = gpd_df.kurtosis()

    pd.testing.assert_series_equal(arctern_kurtosis_series, gpd_kurtosis_series)


def test_mad():
    arctern_mad_series = arctern_df.mad()
    gpd_mad_series = gpd_df.mad()

    pd.testing.assert_series_equal(arctern_mad_series, gpd_mad_series)


def test_max():
    arctern_max_series = arctern_df.max()
    gpd_max_series = gpd_df.max()

    pd.testing.assert_series_equal(arctern_max_series, gpd_max_series)


def test_mean():
    arctern_mean_series = arctern_df.mean()
    gpd_mean_series = gpd_df.mean()

    pd.testing.assert_series_equal(arctern_mean_series, gpd_mean_series)


def test_median():
    arctern_median_series = arctern_df.median()
    gpd_median_series = gpd_df.median()

    pd.testing.assert_series_equal(arctern_median_series, gpd_median_series)


def test_min():
    arctern_min_series = arctern_df.min()
    gpd_min_series = gpd_df.min()

    pd.testing.assert_series_equal(arctern_min_series, gpd_min_series)


def test_mode():
    arctern_mode_df = arctern_df.mode()
    gpd_mode_df = gpd_df.mode()

    pd.testing.assert_frame_equal(arctern_mode_df, gpd_mode_df, check_dtype=False)


def test_pct_change():
    try:
        arctern_pct_change_df = arctern.GeoSeries(df['buildingtext_dropoff']).pct_change()
    except Exception as e1:
        arctern_pct_change_exception = e1
    try:
        gpd_pct_change_df = (
            #gpd.GeoSeries(df['buildingtext_dropoff'].apply(shapely.wkt.loads)).apply(lambda x: x.to_wkb())).pct_change()
            gpd.GeoSeries(df['buildingtext_dropoff'].apply(shapely.wkt.loads)).apply(lambda x: x.wkb)).pct_change()
    except Exception as e2:
        gpd_pct_change_exception = e2

    assert type(arctern_pct_change_exception) == type(gpd_pct_change_exception)


def test_prod():
    arctern_prod_series = arctern_df.prod()
    gpd_prod_series = gpd_df.prod()

    pd.testing.assert_series_equal(arctern_prod_series, gpd_prod_series)


def test_product():
    arctern_product_series = arctern_df.product()
    gpd_product_series = gpd_df.product()

    pd.testing.assert_series_equal(arctern_product_series, gpd_product_series)


def test_quantile():
    arctern_quantile_series = arctern_df.quantile(0.25)
    gpd_quantile_series = gpd_df.quantile(0.25)

    pd.testing.assert_series_equal(arctern_quantile_series, gpd_quantile_series)


def test_rank():
    arctern_rank_df = arctern_df.rank()
    gpd_rank_df = gpd_df.rank()

    pd.testing.assert_frame_equal(arctern_rank_df, gpd_rank_df)


def test_round():
    arctern_round_df = arctern_df.round()
    gpd_round_df = gpd_df.round()

    pd.testing.assert_frame_equal(arctern_round_df, gpd_round_df, check_dtype=False)


def test_sem():
    arctern_sem_series = arctern_df.sem()
    gpd_sem_series = gpd_df.sem()

    pd.testing.assert_series_equal(arctern_sem_series, gpd_sem_series)


def test_skew():
    arctern_skew_series = arctern_df.skew()
    gpd_skew_series = gpd_df.skew()

    pd.testing.assert_series_equal(arctern_skew_series, gpd_skew_series)


def test_sum():
    arctern_sum_series = arctern_df.sum()
    gpd_sum_series = gpd_df.sum()

    pd.testing.assert_series_equal(arctern_sum_series, gpd_sum_series)


def test_std():
    arctern_std_series = arctern_df.std()
    gpd_std_series = gpd_df.std()

    pd.testing.assert_series_equal(arctern_std_series, gpd_std_series)


def test_var():
    arctern_var_series = arctern_df.var()
    gpd_var_series = gpd_df.var()

    pd.testing.assert_series_equal(arctern_var_series, gpd_var_series)


def test_nunique():
    arctern_nunique_series = arctern_df.nunique()
    gpd_nunique_series = gpd_df.nunique()

    pd.testing.assert_series_equal(arctern_nunique_series, gpd_nunique_series)


def test_add_prefix():
    arctern_prefix_df = arctern_df.add_prefix("prefix_")
    gpd_prefix_df = gpd_df.add_prefix("prefix_")

    pd.testing.assert_frame_equal(arctern_prefix_df, gpd_prefix_df, check_dtype=False)


def test_add_suffix():
    arctern_suffix_df = arctern_df.add_suffix("_suffix")
    gpd_suffix_df = gpd_df.add_suffix("_suffix")

    pd.testing.assert_frame_equal(arctern_suffix_df, gpd_suffix_df, check_dtype=False)


def test_align():
    arctern_align_tuple = arctern_df.align(df)
    gpd_align_tuple = gpd_df.align(df)

    pd.testing.assert_frame_equal(arctern_align_tuple[0], gpd_align_tuple[0], check_dtype=False)
    pd.testing.assert_frame_equal(arctern_align_tuple[1], gpd_align_tuple[1], check_dtype=False)


def test_at_time():
    try:
        arctern_at_time_df = arctern_df.at_time("12:00")
    except Exception as e1:
        arctern_at_time_exception = e1
    try:
        gpd_at_time_df = gpd_df.at_time("12:00")
    except Exception as e2:
        gpd_at_time_exception = e2

    assert type(arctern_at_time_exception) == type(gpd_at_time_exception)


def test_between_time():
    try:
        arctern_between_time_df = arctern_df.between_time("12:00")
    except Exception as e1:
        arctern_between_time_exception = e1

    try:
        gpd_between_time_df = gpd_df.between_time("12:00")
    except Exception as e2:
        gpd_between_time_exception = e2

    assert type(arctern_between_time_exception) == type(gpd_between_time_exception)


def test_drop():
    arctern_drop_df = arctern_df.drop(columns=['test_GeoSeries'])
    gpd_drop_df = gpd_df.drop(columns=['test_GeoSeries'])

    pd.testing.assert_frame_equal(arctern_drop_df, gpd_drop_df)


def test_drop_duplicates():
    arctern_drop_duplicates_df = arctern_df.drop_duplicates()
    gpd_drop_duplicates_df = gpd_df.drop_duplicates()

    pd.testing.assert_frame_equal(arctern_drop_duplicates_df, gpd_drop_duplicates_df, check_dtype=False)


def test_duplicates():
    arctern_duplicated_series = arctern_df.duplicated()
    gpd_duplicated_series = gpd_df.duplicated()

    pd.testing.assert_series_equal(arctern_duplicated_series, gpd_duplicated_series)


def test_equals():
    arctern_equal_bool = arctern_df.equals(df)
    gpd_equal_bool = gpd_df.equals(df)

    assert arctern_equal_bool == gpd_equal_bool


def test_filter():
    arctern_filter_df = arctern_df.filter(items=['test_GeoSeries', 'buildingtext_dropoff'])
    gpd_filter_df = gpd_df.filter(items=['test_GeoSeries', 'buildingtext_dropoff'])

    pd.testing.assert_frame_equal(arctern_filter_df, gpd_filter_df, check_dtype=False)


def test_first():
    try:
        arctern_first_df = arctern_df.first("3D")
    except Exception as e1:
        arctern_first_exception = e1
    try:
        gpd_first_df = gpd_df.first("3D")
    except Exception as e2:
        gpd_first_exception = e2

    assert type(arctern_first_exception) == type(gpd_first_exception)


def test_idmax():
    try:
        arctern_idxmax_df = arctern_df.idxmax()
    except Exception as e1:
        arctern_idxmax_exception = e1

    try:
        gpd_idxmax_df = gpd_df.idxmax()
    except Exception as e2:
        gpd_idxmax_exception = e2

    assert type(arctern_idxmax_exception) == type(gpd_idxmax_exception)


def test_idmin():
    try:
        arctern_idxmin_df = arctern_df.idxmin()
    except Exception as e1:
        arctern_idxmin_exception = e1
    try:
        gpd_idxmin_df = gpd_df.idxmin()
    except Exception as e2:
        gpd_idxmin_exception = e2

    assert type(arctern_idxmin_exception) == type(gpd_idxmin_exception)


def test_last():
    try:
        arctern_last_df = arctern_df.last("3D")
    except Exception as e1:
        arctern_last_exception = e1

    try:
        gpd_last_df = gpd_df.last("3D")
    except Exception as e2:
        gpd_last_exception = e2

    assert type(arctern_last_exception) == type(gpd_last_exception)


def test_reindex():
    arctern_reindix_df = arctern_df.reindex(columns=['test_GeoSeries', 'geo'])
    gpd_reindex_df = gpd_df.reindex(columns=['test_GeoSeries', 'geo'])

    pd.testing.assert_frame_equal(arctern_reindix_df, gpd_reindex_df, check_dtype=False)


def test_reindex_like():
    arctern_reindex_like_df = arctern_df.reindex_like(df)
    gpd_reindex_like_df = gpd_df.reindex_like(df)

    pd.testing.assert_frame_equal(arctern_reindex_like_df, gpd_reindex_like_df)


def test_rename():
    arctern_rename_df = arctern_df.rename({'test_GeoSeries': 'geo'})
    gpd_rename_df = gpd_df.rename({'test_GeoSeries': 'geo'})

    pd.testing.assert_frame_equal(arctern_rename_df, gpd_rename_df, check_dtype=False)


def test_rename_axis():
    arctern_rename_axis_df = arctern_df.rename_axis('index', axis='index')
    gpd_rename_axis_df = gpd_df.rename_axis('index', axis='index')

    pd.testing.assert_frame_equal(arctern_rename_axis_df, gpd_rename_axis_df, check_dtype=False)


def test_reset_index():
    arctern_reset_index_df = arctern_df.reset_index()
    gpd_reset_index_df = gpd_df.reset_index()

    pd.testing.assert_frame_equal(arctern_reset_index_df, gpd_reset_index_df, check_dtype=False)


def test_set_axis():
    index_num = len(df.index)
    datetime_index = pd.date_range(start='2020-05-01', periods=index_num, freq='H')
    arctern_set_axis_df = arctern_df.set_axis(datetime_index, axis="index")
    gpd_set_axis_df = gpd_df.set_axis(datetime_index, axis="index")

    pd.testing.assert_frame_equal(arctern_set_axis_df, gpd_set_axis_df, check_dtype=False)


def test_set_index():
    arctern_set_index_df = arctern_df.set_index(['VendorID', 'test_GeoSeries'])
    gpd_set_index_df = gpd_df.set_index(['VendorID', 'test_GeoSeries'])

    pd.testing.assert_frame_equal(arctern_set_index_df, gpd_set_index_df, check_dtype=False)


def test_take():
    arctern_take_df = arctern_df.take([1, 5])
    gpd_take_df = gpd_df.take([1, 5])

    pd.testing.assert_frame_equal(arctern_take_df, gpd_take_df, check_dtype=False)


def test_truncate():
    arctern_truncate_df = arctern_df.truncate(before=2, after=4)
    gpd_truncate_df = gpd_df.truncate(before=2, after=4)

    pd.testing.assert_frame_equal(arctern_truncate_df, gpd_truncate_df, check_dtype=False)


def test_dropna():
    arctern_dropna_df = arctern_df.dropna()
    gpd_dropna_df = gpd_df.dropna()

    pd.testing.assert_frame_equal(arctern_dropna_df, gpd_dropna_df, check_dtype=False)


def test_fillna():
    arctern_fillna_df = arctern_df.fillna('a')
    gpd_fillna_df = gpd_df.fillna('a')

    pd.testing.assert_frame_equal(arctern_fillna_df, gpd_fillna_df, check_dtype=False)


def test_interpolate():
    arctern_interpolate_df = arctern_df.interpolate(method="ffill")
    gpd_interpolate_df = gpd_df.interpolate(method="ffill")

    pd.testing.assert_frame_equal(arctern_interpolate_df, gpd_interpolate_df, check_dtype=False)


def test_droplevel():
    arctern_set_index_df = arctern_df.set_index(['VendorID', 'test_GeoSeries'])
    gpd_set_index_df = gpd_df.set_index(['VendorID', 'test_GeoSeries'])
    arctern_droplevel_df = arctern_set_index_df.droplevel('VendorID')
    gpd_droplevel_df = gpd_set_index_df.droplevel('VendorID')

    pd.testing.assert_frame_equal(arctern_droplevel_df, gpd_droplevel_df, check_dtype=False)


def test_pivot():
    arctern_pivot_df = arctern_df.pivot(index='tpep_pickup_datetime', columns='passenger_count',
                                        values='test_GeoSeries')
    gpd_pivot_df = gpd_df.pivot(index='tpep_pickup_datetime', columns='passenger_count', values='test_GeoSeries')

    pd.testing.assert_frame_equal(arctern_pivot_df, gpd_pivot_df, check_dtype=False)


def test_pivot_table():
    arctern_pivot_table_df = pd.pivot_table(arctern_df, values='test_GeoSeries', index=['tpep_pickup_datetime'],
                                            columns=['passenger_count'], aggfunc=np.sum)
    gpd_pivot_table_df = pd.pivot_table(gpd_df, values='test_GeoSeries', index=['tpep_pickup_datetime'],
                                        columns=['passenger_count'], aggfunc=np.sum)

    pd.testing.assert_frame_equal(arctern_pivot_table_df, gpd_pivot_table_df, check_dtype=False)


def test_reorder_levels():
    try:
        arctern_reorder_levels_df = arctern_df.reorder_levels(order=['VendorID', 'tpep_pickup_datetime'])
    except Exception as e1:
        arctern_reorder_levels_exception = e1
    try:
        gpd_reorder_levels_df = gpd_df.reorder_levels(order=['VendorID', 'tpep_pickup_datetime'])
    except Exception as e2:
        gpd_reorder_levels_exception = e2

    assert type(arctern_reorder_levels_exception) == type(gpd_reorder_levels_exception)


def test_sort_values():
    arctern_sort_values_df = arctern_df.sort_values(by=['tpep_pickup_datetime'])
    gpd_sort_values_df = gpd_df.sort_values(by=['tpep_pickup_datetime'])

    pd.testing.assert_frame_equal(arctern_sort_values_df, gpd_sort_values_df, check_dtype=False)


def test_sort_index():
    arctern_sort_index_df = arctern_df.sort_index()
    gpd_sort_index_df = gpd_df.sort_index()

    pd.testing.assert_frame_equal(arctern_sort_index_df, gpd_sort_index_df, check_dtype=False)


def test_nlargest():
    arctern_nlargest_df = arctern_df.nlargest(5, 'tpep_pickup_datetime')
    gpd_nlargest_df = gpd_df.nlargest(5, 'tpep_pickup_datetime')

    pd.testing.assert_frame_equal(arctern_nlargest_df, gpd_nlargest_df, check_dtype=False)


def test_nsmallest():
    arctern_nsmallest_df = arctern_df.nsmallest(5, 'tpep_pickup_datetime')
    gpd_nsmallest_df = gpd_df.nsmallest(5, 'tpep_pickup_datetime')

    pd.testing.assert_frame_equal(arctern_nsmallest_df, gpd_nsmallest_df, check_dtype=False)


def test_swaplevel():
    try:
        arctern_swaplevel_df = arctern_df.swaplevel()
    except Exception as e1:
        arctern_swaplevel_exception = e1
    try:
        gpd_swaplevel_df = gpd_df.swaplevel()
    except Exception as e2:
        gpd_swaplevel_exception = e2

    assert type(arctern_swaplevel_exception) == type(gpd_swaplevel_exception)


def test_stack():
    arctern_stack_series = arctern_df.stack()
    gpd_stack_series = gpd_df.stack()

    pd.testing.assert_series_equal(arctern_stack_series, gpd_stack_series)


def test_unstack():
    arctern_unstack_series = arctern_df.unstack()
    gpd_unstack_series = gpd_df.unstack()

    pd.testing.assert_series_equal(arctern_unstack_series, gpd_unstack_series)


def test_swapaxes():
    arctern_swapaxes_df = arctern_df.swapaxes(axis1=0, axis2=1)
    gpd_swapaxes_df = gpd_df.swapaxes(axis1=0, axis2=1)

    pd.testing.assert_frame_equal(arctern_swapaxes_df, gpd_swapaxes_df)


def test_melt():
    arctern_melt_df = arctern_df.melt(id_vars=['VendorID'], value_vars=['test_GeoSeries'])
    gpd_melt_df = gpd_df.melt(id_vars=['VendorID'], value_vars=['test_GeoSeries'])

    pd.testing.assert_frame_equal(arctern_melt_df, gpd_melt_df)


def test_explode():
    arctern_explode_df = arctern_df.explode('test_GeoSeries')
    gpd_explode_df = gpd_df.explode('test_GeoSeries')

    pd.testing.assert_frame_equal(arctern_explode_df, gpd_explode_df, check_dtype=False)


def test_squeeze():
    arctern_squeeze_df = arctern_df.squeeze()
    gpd_squeeze_df = gpd_df.squeeze()

    pd.testing.assert_frame_equal(arctern_squeeze_df, gpd_squeeze_df, check_dtype=False)


def test_pow():
    try:
        arctern_pow_df = arctern_df.pow(df)
    except Exception as e1:
        arctern_pow_exception = e1
    try:
        gpd_pow_df = gpd_df.pow(df)
    except Exception as e2:
        gpd_pow_exception = e2

    assert type(arctern_pow_exception) == type(gpd_pow_exception)


def test_dot():
    dot_series = pd.Series([1] * 17,
                           index=["VendorID", "tpep_pickup_datetime", "tpep_dropoff_datetime", "passenger_count",
                                  "trip_distance", "pickup_longitude", "pickup_latitude", "dropoff_longitude",
                                  "dropoff_latitude", "fare_amount", "tip_amount", "total_amount", "buildingid_pickup",
                                  "buildingid_dropoff", "buildingtext_pickup", "buildingtext_dropoff",
                                  "test_GeoSeries"])
    try:
        arctern_dot_series = arctern_df.dot(dot_series)
    except Exception as e1:
        arctern_dot_exception = e1
    try:
        gpd_dot_deries = gpd_df.dot(dot_series)
    except Exception as e2:
        gpd_dot_exception = e2

    assert type(arctern_dot_exception) == type(gpd_dot_exception)


def test_radd():
    try:
        arctern_radd_series = arctern_df.radd(df)
    except Exception as e1:
        arctern_radd_exception = e1
    try:
        gpd_add_deries = gpd_df.radd(df)
    except Exception as e2:
        gpd_radd_exception = e2

    assert type(arctern_radd_exception) == type(gpd_radd_exception)


def test_rsub():
    try:
        arctern_rsub_series = arctern_df.rsub(df)
    except Exception as e1:
        arctern_rsub_exception = e1
    try:
        gpd_rsub_deries = gpd_df.rsub(df)
    except Exception as e2:
        gpd_rsub_exception = e2

    assert type(arctern_rsub_exception) == type(gpd_rsub_exception)


def test_rmul():
    try:
        arctern_rmul_series = arctern_df.rmul(df)
    except Exception as e1:
        arctern_rmul_exception = e1
    try:
        gpd_rmul_deries = gpd_df.rmul(df)
    except Exception as e2:
        gpd_rmul_exception = e2

    assert type(arctern_rmul_exception) == type(gpd_rmul_exception)


def test_rdiv():
    try:
        arctern_rdiv_series = arctern_df.rdiv(df)
    except Exception as e1:
        arctern_rdiv_exception = e1
    try:
        gpd_rdiv_deries = gpd_df.rdiv(df)
    except Exception as e2:
        gpd_rdiv_exception = e2

    assert type(arctern_rdiv_exception) == type(gpd_rdiv_exception)


def test_rtruediv():
    try:
        arctern_rtruediv_series = arctern_df.rtruediv(df)
    except Exception as e1:
        arctern_rtruediv_exception = e1
    try:
        gpd_rtruediv_deries = gpd_df.rtruediv(df)
    except Exception as e2:
        gpd_rtruediv_exception = e2

    assert type(arctern_rtruediv_exception) == type(gpd_rtruediv_exception)


def test_rfloor():
    try:
        arctern_rfloordiv_series = arctern_df.rfloordiv(df)
    except Exception as e1:
        arctern_rfloordiv_exception = e1
    try:
        gpd_rfloordiv_deries = gpd_df.rfloordiv(df)
    except Exception as e2:
        gpd_rfloordiv_exception = e2

    assert type(arctern_rfloordiv_exception) == type(gpd_rfloordiv_exception)


def test_T():
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.transpose.html
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.T.html
    geo_s = GeoSeries(geo_dropoff.to_list())
    pd_s = pd.Series(geo_dropoff.to_list())
    pd_s_wkb = trans2wkb4series(pd_s)
    pd.testing.assert_series_equal(geo_s, pd_s_wkb, check_dtype=False)

    geo_df = pd.DataFrame({'s': geo_s})
    pd_df_wkb = pd.DataFrame({'s': pd_s_wkb})
    pd.testing.assert_frame_equal(geo_df, pd_df_wkb, check_dtype=False)

    geo_res = geo_df.T
    pd_res = pd_df_wkb.T

    pd.testing.assert_frame_equal(geo_res, pd_res, check_dtype=False)


def test_append():
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.append.html
    geo_s_l = GeoSeries(geo_dropoff.to_list())
    geo_df_l = pd.DataFrame({'lkey': geo_s_l})
    geo_s_r = GeoSeries(geo_pickup.to_list())
    geo_df_r = pd.DataFrame({'rkey': geo_s_r})

    pd_s_l = pd.Series(geo_dropoff.to_list())
    pd_s_l_wkb = trans2wkb4series(pd_s_l)
    pd_df_l = pd.DataFrame({'lkey': pd_s_l_wkb})
    pd_s_r = pd.Series(geo_pickup.to_list())
    pd_s_r_wkb = trans2wkb4series(pd_s_r)
    pd_df_r = pd.DataFrame({'rkey': pd_s_r_wkb})

    pd.testing.assert_frame_equal(geo_df_l, pd_df_l, check_dtype=False)
    pd.testing.assert_frame_equal(geo_df_r, pd_df_r, check_dtype=False)

    geo_res = geo_df_l.append(geo_df_r)
    pd_res = pd_df_l.append(pd_df_r)
    pd.testing.assert_frame_equal(geo_res, pd_res, check_dtype=False)


def test_assign():
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.assign.html
    geo_s = GeoSeries(geo_dropoff.to_list())
    pd_s = pd.Series(geo_dropoff.to_list())
    pd_s_wkb = trans2wkb4series(pd_s)
    pd.testing.assert_series_equal(geo_s, pd_s_wkb, check_dtype=False)

    geo_df = pd.DataFrame({'s': geo_s})
    pd_df_wkb = pd.DataFrame({'s': pd_s_wkb})
    pd.testing.assert_frame_equal(geo_df, pd_df_wkb, check_dtype=False)

    geo_res = geo_df.assign(temp_f=lambda x: x.s)
    pd_res = pd_df_wkb.assign(temp_f=lambda x: x.s)

    pd.testing.assert_frame_equal(geo_res, pd_res, check_dtype=False)


def test_join():
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.join.html
    geo_s_l = GeoSeries(geo_dropoff.to_list())
    geo_df_l = pd.DataFrame({'lkey': geo_s_l})
    geo_s_r = GeoSeries(geo_pickup.to_list())
    geo_df_r = pd.DataFrame({'rkey': geo_s_r})

    pd_s_l = pd.Series(geo_dropoff.to_list())
    pd_s_l_wkb = trans2wkb4series(pd_s_l)
    pd_df_l = pd.DataFrame({'lkey': pd_s_l_wkb})
    pd_s_r = pd.Series(geo_pickup.to_list())
    pd_s_r_wkb = trans2wkb4series(pd_s_r)
    pd_df_r = pd.DataFrame({'rkey': pd_s_r_wkb})

    pd.testing.assert_frame_equal(geo_df_l, pd_df_l, check_dtype=False)
    pd.testing.assert_frame_equal(geo_df_r, pd_df_r, check_dtype=False)

    geo_res = geo_df_l.join(geo_df_r)
    pd_res = pd_df_l.join(pd_df_r)
    pd.testing.assert_frame_equal(geo_res, pd_res, check_dtype=False)


def test_merge():
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.merge.html
    geo_s_old = GeoSeries(geo_dropoff.to_list())
    geo_df_old = pd.DataFrame({'lkey': geo_s_old})
    geo_s_new = GeoSeries(geo_pickup.to_list())
    geo_df_new = pd.DataFrame({'rkey': geo_s_new})

    pd_s_old = pd.Series(geo_dropoff.to_list())
    pd_s_old_wkb = trans2wkb4series(pd_s_old)
    pd_df_old = pd.DataFrame({'lkey': pd_s_old_wkb})
    pd_s_new = pd.Series(geo_pickup.to_list())
    pd_s_new_wkb = trans2wkb4series(pd_s_new)
    pd_df_new = pd.DataFrame({'rkey': pd_s_new_wkb})

    pd.testing.assert_frame_equal(geo_df_old, pd_df_old, check_dtype=False)
    pd.testing.assert_frame_equal(geo_df_new, pd_df_new, check_dtype=False)

    geo_res = geo_df_old.merge(geo_df_new, left_on='lkey', right_on='rkey', suffixes=('_left', '_right'))
    pd_res = pd_df_old.merge(pd_df_new, left_on='lkey', right_on='rkey', suffixes=('_left', '_right'))

    pd.testing.assert_frame_equal(geo_res, pd_res, check_dtype=False)


def test_update():
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.update.html
    geo_s_old = GeoSeries(geo_dropoff.to_list())
    geo_df_old = pd.DataFrame(geo_s_old)
    geo_s_new = GeoSeries(geo_pickup.to_list())
    geo_df_new = pd.DataFrame(geo_s_new)

    pd_s_old = pd.Series(geo_dropoff.to_list())
    pd_s_old_wkb = trans2wkb4series(pd_s_old)
    pd_df_old = pd.DataFrame(pd_s_old_wkb)
    pd_s_new = pd.Series(geo_pickup.to_list())
    pd_s_new_wkb = trans2wkb4series(pd_s_new)
    pd_df_new = pd.DataFrame(pd_s_new_wkb)

    geo_df_old.update(geo_df_new)
    pd_df_old.update(pd_df_new)

    pd.testing.assert_frame_equal(geo_df_old, pd_df_old, check_dtype=False)
    pd.testing.assert_frame_equal(geo_df_new, pd_df_new, check_dtype=False)


def test_asfreq():
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.asfreq.html
    index = pd.date_range('1/1/2000', periods=10, freq='T')
    geo_series = GeoSeries(geo_dropoff.to_list(), index=index)
    geo_df = pd.DataFrame({'geo_dropoff': geo_series})
    pd_series = GeoSeries(geo_dropoff.to_list(), index=index)
    pd_df = pd.DataFrame({'geo_dropoff': pd_series})

    geo_res1 = geo_df.asfreq(freq='30S')
    pd_res1 = pd_df.asfreq(freq='30S')
    assert pd_res1.equals(geo_res1)

    geo_res2 = geo_df.asfreq(freq='30S', fill_value=None)
    pd_res2 = pd_df.asfreq(freq='30S', fill_value=None)
    assert pd_res2.equals(geo_res2)

    geo_res3 = geo_df.asfreq(freq='30S', method='bfill')
    pd_res3 = pd_df.asfreq(freq='30S', method='bfill')
    assert pd_res3.equals(geo_res3)


def test_asof():
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.asof.html
    geo_s = GeoSeries(geo_dropoff.to_list(), index=range(0, len(geo_dropoff)))
    pd_s = pd.Series(geo_dropoff.to_list(), index=range(0, len(geo_dropoff)))
    geo_res = geo_s.asof(3)
    pd_res = pd_s.asof(3)
    import pygeos
    assert pygeos.to_wkb(pygeos.Geometry(pd_res)) == geo_res


def test_shift():
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.shift.html
    geo_s = GeoSeries(geo_dropoff.to_list(), index=range(0, len(geo_dropoff)))
    pd_s = pd.Series(geo_dropoff.to_list(), index=range(0, len(geo_dropoff)))

    geo_df = pd.DataFrame({'s': geo_s})
    pd_df = pd.DataFrame({'s': pd_s})
    pd_df_wkb = trans2wkb(pd_df, 's')
    pd.testing.assert_frame_equal(geo_df, pd_df_wkb, check_dtype=False)

    geo_res1 = geo_df.shift(periods=1)
    pd_res1 = pd_df_wkb.shift(periods=1)
    pd.testing.assert_frame_equal(geo_res1, pd_res1, check_dtype=False)

    # TBD : https://github.com/zilliztech/arctern/issues/645
    # geo_res2 = geo_df.shift(periods=1, axis=1)
    # pd_res2 = pd_df_wkb.shift(periods=1, axis=1)
    # pd.testing.assert_frame_equal(geo_res2,pd_res2,check_dtype=False)


def test_slice_shift():
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.slice_shift.html
    geo_s = GeoSeries(geo_dropoff.to_list(), index=range(0, len(geo_dropoff)))
    pd_s = pd.Series(geo_dropoff.to_list(), index=range(0, len(geo_dropoff)))

    geo_df = pd.DataFrame({'s': geo_s})
    pd_df = pd.DataFrame({'s': pd_s})
    pd_df_wkb = trans2wkb(pd_df, 's')
    pd.testing.assert_frame_equal(geo_df, pd_df_wkb, check_dtype=False)

    geo_res1 = geo_df.slice_shift(periods=1)
    pd_res1 = pd_df_wkb.slice_shift(periods=1)
    pd.testing.assert_frame_equal(geo_res1, pd_res1, check_dtype=False)

    geo_res1 = geo_df.slice_shift(periods=1, axis='columns')
    pd_res1 = pd_df_wkb.slice_shift(periods=1, axis='columns')
    pd.testing.assert_frame_equal(geo_res1, pd_res1, check_dtype=False)


def test_tshift():
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.tshift.html
    index = pd.date_range('1/1/2000', periods=10, freq='T')

    geo_s = GeoSeries(geo_dropoff.to_list(), index=index)
    geo_df = pd.DataFrame({'s': geo_s})

    pd_s = pd.Series(geo_dropoff.to_list(), index=index, dtype=object)
    pd_df = pd.DataFrame({'s': pd_s})
    pd_df_wkb = trans2wkb(pd_df, 's', index)

    geo_res1 = geo_df.tshift(freq='30S', axis=0)
    pd_res1 = pd_df_wkb.tshift(freq='30S', axis=0)
    pd.testing.assert_frame_equal(geo_res1, pd_res1, check_dtype=False)

    geo_res2 = geo_df.tshift(freq='30S', periods=1)
    pd_res2 = pd_df_wkb.tshift(freq='30S', periods=1)

    pd.testing.assert_frame_equal(geo_res2, pd_res2, check_dtype=False)


def test_first_valid_index():
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.first_valid_index.html
    index = pd.date_range('1/1/2000', periods=10, freq='T')

    geo_s = GeoSeries(geo_dropoff.to_list(), index=index)
    geo_df = pd.DataFrame({'s': geo_s})

    pd_s = pd.Series(geo_dropoff.to_list(), index=index, dtype=object)
    pd_df = pd.DataFrame({'s': pd_s})
    pd_df_wkb = trans2wkb(pd_df, 's', index)

    geo_res1 = geo_df.first_valid_index()
    pd_res1 = pd_df_wkb.first_valid_index()
    assert pd_res1 == geo_res1


def test_last_valid_index():
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.last_valid_index.html
    index = pd.date_range('1/1/2000', periods=10, freq='T')

    geo_s = GeoSeries(geo_dropoff.to_list(), index=index)
    geo_df = pd.DataFrame({'s': geo_s})

    pd_s = pd.Series(geo_dropoff.to_list(), index=index, dtype=object)
    pd_df = pd.DataFrame({'s': pd_s})
    pd_df_wkb = trans2wkb(pd_df, 's', index)

    geo_res1 = geo_df.last_valid_index()
    pd_res1 = pd_df_wkb.last_valid_index()
    assert pd_res1 == geo_res1


def test_resample():
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.resample.html
    index = pd.date_range('1/1/2000', periods=10, freq='T')

    geo_s = GeoSeries(geo_dropoff.to_list(), index=index)
    geo_df = pd.DataFrame({'s': geo_s})

    pd_s = pd.Series(geo_dropoff.to_list(), index=index, dtype=object)
    pd_df = pd.DataFrame({'s': pd_s})
    pd_df_wkb = trans2wkb(pd_df, 's', index)

    geo_res1 = geo_df.resample('3T').pad()[0:4]
    pd_res1 = pd_df_wkb.resample('3T').pad()[0:4]
    pd.testing.assert_frame_equal(geo_res1, pd_res1, check_dtype=False)

    geo_res2 = geo_df.resample('3T', label='right', closed='right').first().count()
    pd_res2 = pd_df_wkb.resample('3T', label='right', closed='right').first().count()
    assert (geo_res2 == pd_res2).all()


def test_to_period():
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_period.html
    index = pd.date_range('1/1/2000', periods=10, freq='T')

    geo_s = GeoSeries(geo_dropoff.to_list(), index=index)
    geo_df = pd.DataFrame({'s': geo_s})

    pd_s = pd.Series(geo_dropoff.to_list(), index=index, dtype=object)
    pd_df = pd.DataFrame({'s': pd_s})
    pd_df_wkb = trans2wkb(pd_df, 's', index)

    geo_res1 = geo_df.to_period(copy=False, axis=0, freq='30T')
    pd_res1 = pd_df_wkb.to_period(copy=False, axis=0, freq='30T')
    pd.testing.assert_frame_equal(geo_res1, pd_res1, check_dtype=False)


def test_tz_convert():
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.tz_convert.html
    index = pd.date_range('1/1/2000', periods=10, freq='T', tz='US/Central')

    geo_s = GeoSeries(geo_dropoff.to_list(), index=index)
    geo_df = pd.DataFrame({'s': geo_s})

    pd_s = pd.Series(geo_dropoff.to_list(), index=index, dtype=object)
    pd_df = pd.DataFrame({'s': pd_s})
    pd_df_wkb = trans2wkb(pd_df, 's', index)

    geo_res1 = geo_df.tz_convert(tz='Europe/Berlin')
    pd_res1 = pd_df_wkb.tz_convert(tz='Europe/Berlin')
    pd.testing.assert_frame_equal(geo_res1, pd_res1, check_dtype=False)


def test_tz_localize():
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.tz_localize.html
    index = pd.date_range('1/1/2000', periods=10, freq='T')

    geo_s = GeoSeries(geo_dropoff.to_list(), index=index)
    geo_df = pd.DataFrame({'s': geo_s})

    pd_s = pd.Series(geo_dropoff.to_list(), index=index, dtype=object)
    pd_df = pd.DataFrame({'s': pd_s})
    pd_df_wkb = trans2wkb(pd_df, 's', index)

    geo_res1 = geo_df.tz_localize('CET')
    pd_res1 = pd_df_wkb.tz_localize('CET')
    pd.testing.assert_frame_equal(geo_res1, pd_res1, check_dtype=False)

    geo_res2 = geo_df.tz_localize('CET', ambiguous='infer')
    pd_res2 = pd_df_wkb.tz_localize('CET', ambiguous='infer')
    pd.testing.assert_frame_equal(geo_res2, pd_res2, check_dtype=False)

    geo_res3 = geo_df.tz_localize('CET', ambiguous=np.zeros(10))
    pd_res3 = pd_df_wkb.tz_localize('CET', ambiguous=np.zeros(10))
    pd.testing.assert_frame_equal(geo_res3, pd_res3, check_dtype=False)


def test_attrs():
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.attrs.html
    index = pd.date_range('1/1/2000', periods=10, freq='T')

    geo_s = GeoSeries(geo_dropoff.to_list(), index=index)
    geo_df = pd.DataFrame({'s': geo_s})

    pd_s = pd.Series(geo_dropoff.to_list(), index=index, dtype=object)
    pd_df = pd.DataFrame({'s': pd_s})
    pd_df_wkb = trans2wkb(pd_df, 's', index)

    geo_res1 = geo_df.attrs
    pd_res1 = pd_df_wkb.attrs
    assert geo_res1 == pd_res1


def test_to_dense():
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.sparse.to_dense.html
    pd_s = pd.Series(["POINT(1 1)", None, None, None, None, None])
    pd_s_wkb = trans2wkb4series(pd_s)
    pd_df = pd.DataFrame({"s": pd.arrays.SparseArray(pd_s_wkb)})
    geo_df = pd.DataFrame({"s": pd.arrays.SparseArray(GeoSeries(["POINT(1 1)", None, None, None, None, None]))})
    pd_res = pd_df.sparse.to_dense()
    geo_res = geo_df.sparse.to_dense()
    pd.testing.assert_frame_equal(geo_res, pd_res, check_dtype=False)


def test_info():
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.info.html
    geo_s = GeoSeries(geo_dropoff.to_list())
    pd_s = pd.Series(geo_dropoff.to_list())
    pd_s_wkb = trans2wkb4series(pd_s)
    pd.testing.assert_series_equal(geo_s.astype(object), pd_s_wkb.astype(object), check_dtype=False)  # (as expected)

    geo_df = pd.DataFrame({'val': geo_s})
    pd_df = pd.DataFrame({'val': pd_s_wkb})
    pd.testing.assert_frame_equal(geo_df, pd_df, check_dtype=False)

    geo_df.info()
    pd_df.info()


#if 0:
#    def test_to_parquet():
#        # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_parquet.html
#        geo_s = GeoSeries(geo_dropoff.to_list())
#        pd_s = pd.Series(geo_dropoff.to_list())
#        pd_s_wkb = trans2wkb4series(pd_s)
#        pd.testing.assert_series_equal(geo_s.astype(object), pd_s_wkb.astype(object),
#                                       check_dtype=False)  # (as expected)

#        geo_df = pd.DataFrame({'val': geo_s})
#        pd_df = pd.DataFrame({'val': pd_s_wkb})
#        pd.testing.assert_frame_equal(geo_df, pd_df, check_dtype=False)

#        geo_res1 = geo_df.to_parquet('geo.parquet.gzip', compression='gzip')
#        pd_res1 = pd_df.to_parquet('pd.parquet.gzip', compression='gzip')

#        read_geo_res1 = pd.read_parquet('geo.parquet.gzip')
#        read_pd_res1 = pd.read_parquet('pd.parquet.gzip')

#        assert (read_geo_res1 == read_pd_res1).all().all()

#if 0:
#    def test_to_pickle():
#        # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_pickle.html
#        geo_s = GeoSeries(geo_dropoff.to_list())
#        pd_s = pd.Series(geo_dropoff.to_list())
#        pd_s_wkb = trans2wkb4series(pd_s)
#        pd.testing.assert_series_equal(geo_s.astype(object), pd_s_wkb.astype(object),
#                                       check_dtype=False)  # (as expected)

#        geo_df = pd.DataFrame({'val': geo_s})
#        pd_df = pd.DataFrame({'val': pd_s_wkb})
#        pd.testing.assert_frame_equal(geo_df, pd_df, check_dtype=False)

#        geo_res1 = geo_df.to_pickle("geo.pkl")
#        pd_res1 = pd_df.to_pickle("pd.pkl")

#        read_geo_res1 = pd.read_pickle("geo.pkl")
#        read_pd_res1 = pd.read_pickle("pd.pkl")

#        assert (read_geo_res1 == read_pd_res1).all().all()


def test_to_csv():
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_csv.html
    geo_s = GeoSeries(geo_dropoff.to_list())
    pd_s = pd.Series(geo_dropoff.to_list())
    pd_s_wkb = trans2wkb4series(pd_s)
    pd.testing.assert_series_equal(geo_s.astype(object), pd_s_wkb.astype(object), check_dtype=False)  # (as expected)

    geo_df = pd.DataFrame({'val': geo_s})
    pd_df = pd.DataFrame({'val': pd_s_wkb})
    pd.testing.assert_frame_equal(geo_df, pd_df, check_dtype=False)

    geo_res1 = geo_df.to_csv()
    pd_res1 = pd_df.to_csv()
    assert geo_res1 == pd_res1


def test_dict():
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_dict.html
    geo_s = GeoSeries(geo_dropoff.to_list())
    pd_s = pd.Series(geo_dropoff.to_list())
    pd_s_wkb = trans2wkb4series(pd_s)
    pd.testing.assert_series_equal(geo_s.astype(object), pd_s_wkb.astype(object), check_dtype=False)  # (as expected)

    geo_df = pd.DataFrame({'val': geo_s})
    pd_df = pd.DataFrame({'val': pd_s_wkb})
    pd.testing.assert_frame_equal(geo_df, pd_df, check_dtype=False)

    geo_res1 = geo_df.to_dict()
    pd_res1 = pd_df.to_dict()
    assert geo_res1 == pd_res1

    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.from_dict.html
    assert (pd.DataFrame.from_dict(geo_res1) == pd.DataFrame.from_dict(pd_res1)).all().all()


def test_to_json():
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_json.html
    # pandas error when use geo_dropoff...
    geo_s = GeoSeries(["POINT (9 0)"])
    pd_s = pd.Series(["POINT (9 0)"])
    # geo_s = GeoSeries(geo_dropoff.to_list())
    # pd_s = pd.Series(geo_dropoff.to_list())
    pd_s_wkb = trans2wkb4series(pd_s)
    pd.testing.assert_series_equal(geo_s.astype(object), pd_s_wkb.astype(object), check_dtype=False)  # (as expected)

    geo_df = pd.DataFrame({'val': geo_s})
    pd_df = pd.DataFrame({'val': pd_s_wkb})
    pd.testing.assert_frame_equal(geo_df, pd_df, check_dtype=False)

    geo_res1 = geo_df.to_json(orient='table')
    pd_res1 = pd_df.to_json(orient='table')
    assert geo_res1 == pd_res1

    geo_res2 = geo_df.to_json(orient='split')
    pd_res2 = pd_df.to_json(orient='split')
    assert geo_res2 == pd_res2

    geo_res3 = geo_df.to_json(orient='records')
    pd_res3 = pd_df.to_json(orient='records')
    assert geo_res3 == pd_res3

    geo_res4 = geo_df.to_json(orient='index')
    pd_res4 = pd_df.to_json(orient='index')
    assert geo_res4 == pd_res4

    geo_res5 = geo_df.to_json(orient='columns')
    pd_res5 = pd_df.to_json(orient='columns')
    assert geo_res5 == pd_res5

    geo_res6 = geo_df.to_json(orient='values')
    pd_res6 = pd_df.to_json(orient='values')
    assert geo_res6 == pd_res6


def test_to_html():
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_html.html
    geo_s = GeoSeries(geo_dropoff.to_list())
    pd_s = pd.Series(geo_dropoff.to_list())
    pd_s_wkb = trans2wkb4series(pd_s)
    pd.testing.assert_series_equal(geo_s.astype(object), pd_s_wkb.astype(object), check_dtype=False)  # (as expected)

    geo_df = pd.DataFrame({'val': geo_s})
    pd_df = pd.DataFrame({'val': pd_s})
    pd_df_wkb = pd.DataFrame({'val': pd_s_wkb})
    pd.testing.assert_frame_equal(geo_df, pd_df_wkb, check_dtype=False)

    geo_res1 = geo_df.to_html()
    pd_res1 = pd_df.to_html()
    assert geo_res1 == pd_res1


def test_to_latex():
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_latex.html
    geo_s = GeoSeries(geo_dropoff.to_list())
    pd_s = pd.Series(geo_dropoff.to_list())
    pd_s_wkb = trans2wkb4series(pd_s)
    pd.testing.assert_series_equal(geo_s.astype(object), pd_s_wkb.astype(object), check_dtype=False)  # (as expected)

    geo_df = pd.DataFrame({'val': geo_s})
    pd_df = pd.DataFrame({'val': pd_s})
    pd_df_wkb = pd.DataFrame({'val': pd_s_wkb})
    pd.testing.assert_frame_equal(geo_df, pd_df_wkb, check_dtype=False)

    geo_res1 = geo_df.to_latex()
    pd_res1 = pd_df.to_latex()
    pd_res1_wkb = pd_df_wkb.to_latex()
    assert geo_res1 == pd_res1


def test_to_records():
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_records.html
    geo_s = GeoSeries(geo_dropoff.to_list())
    pd_s = pd.Series(geo_dropoff.to_list())
    pd_s_wkb = trans2wkb4series(pd_s)
    pd.testing.assert_series_equal(geo_s.astype(object), pd_s_wkb.astype(object), check_dtype=False)  # (as expected)

    geo_df = pd.DataFrame({'val': geo_s})
    pd_df = pd.DataFrame({'val': pd_s})
    pd_df_wkb = pd.DataFrame({'val': pd_s_wkb})
    pd.testing.assert_frame_equal(geo_df, pd_df_wkb, check_dtype=False)

    geo_res1 = geo_df.to_records()
    pd_res1 = pd_df_wkb.to_records()
    assert (geo_res1 == pd_res1).all()

    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.from_records.html
    assert (pd.DataFrame.from_records(geo_res1) == pd.DataFrame.from_records(pd_res1)).all().all()


def test_to_string():
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_string.html
    geo_s = GeoSeries(["POINT (9 0)", "POLYGON ((1 1,1 2,2 2,1 1))"])
    pd_s = pd.Series(["POINT (9 0)", "POLYGON ((1 1,1 2,2 2,1 1))"])
    # geo_s = GeoSeries(geo_dropoff.to_list())
    # pd_s = pd.Series(geo_dropoff.to_list())
    pd_s_wkb = trans2wkb4series(pd_s)
    pd.testing.assert_series_equal(geo_s.astype(object), pd_s_wkb.astype(object), check_dtype=False)  # (as expected)

    geo_df = pd.DataFrame({'val': geo_s})
    pd_df = pd.DataFrame({'val': pd_s})
    pd_df_wkb = pd.DataFrame({'val': pd_s_wkb})
    pd.testing.assert_frame_equal(geo_df, pd_df_wkb, check_dtype=False)

    geo_res1 = geo_df.to_string()
    pd_res1 = pd_df.to_string()
    assert geo_res1 == pd_res1


#def test_style():
#    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.style.html
#    geo_s = GeoSeries(geo_dropoff.to_list())
#    pd_s = pd.Series(geo_dropoff.to_list())
#    pd_s_wkb = trans2wkb4series(pd_s)
#    pd.testing.assert_series_equal(geo_s.astype(object), pd_s_wkb.astype(object), check_dtype=False)  # (as expected)

#    geo_df = pd.DataFrame({'val': geo_s})
#    pd_df = pd.DataFrame({'val': pd_s})
#    pd_df_wkb = pd.DataFrame({'val': pd_s_wkb})
#    pd.testing.assert_frame_equal(geo_df, pd_df_wkb, check_dtype=False)

#    geo_res1 = geo_df.style
#    pd_res1 = pd_df.style
#    assert geo_res1.data.to_string() == pd_res1.data.to_string()

def test_replace():
    arctern_replace_df = arctern_df.replace("POLYGON ((-73.9833003295812 40.7590607716671,-73.983284516568 40.7590540993346,-73.9831103296997 40.7589805990397,-73.9832739116023 40.7587564485334,-73.9832848295013 40.7587414900311,-73.983320325137 40.75875646787,-73.983664080401 40.7589015182181,-73.983623439378 40.7589572077704,-73.9835486408108 40.7590597024757,-73.9834895815413 40.7591406286961,-73.9834880812315 40.7591399954262,-73.9833003295812 40.7590607716671))", "POLYGON((1 1, 2 2, 3 3))")
    gpd_replace_df = gpd_df.replace("POLYGON ((-73.9833003295812 40.7590607716671,-73.983284516568 40.7590540993346,-73.9831103296997 40.7589805990397,-73.9832739116023 40.7587564485334,-73.9832848295013 40.7587414900311,-73.983320325137 40.75875646787,-73.983664080401 40.7589015182181,-73.983623439378 40.7589572077704,-73.9835486408108 40.7590597024757,-73.9834895815413 40.7591406286961,-73.9834880812315 40.7591399954262,-73.9833003295812 40.7590607716671))", "POLYGON((1 1, 2 2, 3 3))")

    pd.testing.assert_frame_equal(arctern_replace_df, gpd_replace_df, check_dtype=False)
