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

# pylint: disable=too-many-lines,redefined-outer-name,bare-except

import pytest
import pandas as pd
from arctern import GeoSeries
import arctern

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


nyc_df = pd.read_csv("/home/czs/nyc_taxi.csv",
                     dtype=nyc_schema,
                     date_parser=pd.to_datetime,
                     parse_dates=["tpep_pickup_datetime", "tpep_dropoff_datetime"])

geo_dropoff = nyc_df['buildingtext_dropoff'].dropna().head(10)
geo_pickup = nyc_df['buildingtext_pickup'].dropna().head(10)


@pytest.fixture()
def geo_s():
    return GeoSeries(geo_dropoff.to_list())

@pytest.fixture()
def pd_s():
    x = pd.Series(geo_dropoff.to_list())
    return trans2wkb4series(x, x.index)

def test_equals(geo_s, pd_s):
    assert not geo_s.equals(pd_s)

@pytest.mark.skip("not support first")
def test_first():
    pass

@pytest.mark.skip("not support last")
def test_last():
    pass

def test_head(geo_s, pd_s):
    half = geo_s.count() // 2
    ret1 = geo_s.head(half)
    ret2 = pd_s.head(half)
    pd.testing.assert_series_equal(ret1, ret2, check_dtype=False)

@pytest.mark.skip("not support idmax")
def test_idmax():
    pass


@pytest.mark.skip("not support idmin")
def test_idmin():
    pass


def test_isin(geo_s, pd_s):
    ret = geo_s.isin(list(pd_s[::]))
    assert all(ret)

def test_reindex(geo_s, pd_s):
    count = geo_s.count()
    new_index = ['index_%d'%i for i in range(count)]
    ret1 = geo_s.reindex(new_index)
    ret2 = pd_s.reindex(new_index)
    pd.testing.assert_series_equal(ret1, ret2, check_dtype=False)


def test_reindex_like(geo_s, pd_s):
    count = geo_s.count()
    index_ = ["index_%d"%i for i in range(count)]
    geo_s.index = index_
    pd_s.index = index_

    sr2 = pd.Series(range(count-1))
    index2_ = ["index_%d"%i for i in range(count-1)]
    sr2.index = index2_

    ret1 = geo_s.reindex_like(sr2)
    ret2 = pd_s.reindex_like(sr2)

    pd.testing.assert_series_equal(ret1, ret2, check_dtype=False)


def test_rename(geo_s, pd_s):
    geo_s.rename("test_1", inplace=True)
    pd_s.rename("test_1", inplace=True)
    pd.testing.assert_series_equal(geo_s, pd_s, check_dtype=False)

def test_rename_axis(geo_s, pd_s):
    geo_s.rename_axis("test_1", inplace=True)
    pd_s.rename_axis("test_1", inplace=True)
    pd.testing.assert_series_equal(geo_s, pd_s, check_dtype=False)


def test_reset_index(geo_s, pd_s):
    count = geo_s.count()
    new_index = ["index_%d"%i for i in range(count)]
    geo_s.index = new_index
    pd_s.index = new_index
    geo_s = geo_s.reset_index()
    pd_s = pd_s.reset_index()
    pd.testing.assert_frame_equal(geo_s, pd_s, check_dtype=False)


def test_sample(geo_s):
    ret = geo_s.sample(frac=0.5, replace=True, random_state=1)
    assert ret.count() == 4


def test_set_axis(geo_s, pd_s):
    count = geo_s.count()
    new_indexs = ['a'] * count
    geo_s.set_axis(new_indexs)
    pd_s.set_axis(new_indexs)
    ret1 = geo_s.index
    ret2 = pd_s.index
    pd.testing.assert_index_equal(ret1, ret2)

def test_take(geo_s, pd_s):
    geo_s = geo_s.take([0, -1])
    pd_s = geo_s.take([0, -1])
    pd.testing.assert_series_equal(geo_s, pd_s, check_dtype=False)

def test_tail(geo_s, pd_s):
    geo_s = geo_s.tail(-1)
    pd_s = pd_s.tail(-1)
    pd.testing.assert_series_equal(geo_s, pd_s, check_dtype=False)

def test_truncate(geo_s, pd_s):
    before = 2
    after = geo_s.count()
    geo_s = geo_s.truncate(before=before, after=after)
    pd_s = pd_s.truncate(before=before, after=after)
    pd.testing.assert_series_equal(geo_s, pd_s, check_dtype=False)


def test_where(geo_s, pd_s):
    geo_s = geo_s.where(geo_s.npoints <= 10)
    pd_s = pd_s.where(geo_s.npoints <= 10)
    pd.testing.assert_series_equal(geo_s, pd_s, check_dtype=False)


def test_mask(geo_s):
    ret1 = geo_s.mask(geo_s.npoints < 10)
    ret2 = geo_s.mask(geo_s.npoints >= 10)
    assert geo_s.count() == (ret1.count() + ret2.count())


def test_add_prefix(geo_s, pd_s):
    geo_s = geo_s.add_prefix("prefix_")
    pd_s = pd_s.add_prefix("prefix_")
    pd.testing.assert_series_equal(geo_s, pd_s, check_dtype=False)


def test_add_suffix(geo_s, pd_s):
    geo_s = geo_s.add_suffix("_suffix")
    pd_s = pd_s.add_suffix("_suffix")
    pd.testing.assert_series_equal(geo_s, pd_s, check_dtype=False)


def test_filter(geo_s, pd_s):
    total = geo_s.count()
    index = pd.Index(["%d_suffix" % i if i % 2 else i for i in range(total)], dtype='object')

    geo_s.index = index
    pd_s.index = index

    ret1 = geo_s.filter(regex='_suffix$', axis=0)
    ret2 = pd_s.filter(regex='_suffix$', axis=0)
    pd.testing.assert_series_equal(ret1, ret2, check_dtype=False)


def test_isna(geo_s, pd_s):
    ret1 = geo_s.isna()
    ret2 = pd_s.isna()
    pd.testing.assert_series_equal(ret1, ret2, check_dtype=False)


def test_notna(geo_s, pd_s):
    ret1 = geo_s.notna()
    ret2 = pd_s.notna()
    pd.testing.assert_series_equal(ret1, ret2, check_dtype=False)


def test_dropna(geo_s, pd_s):
    ret1 = geo_s.dropna()
    ret2 = pd_s.dropna()
    pd.testing.assert_series_equal(ret1, ret2, check_dtype=False)


def test_fillna(geo_s, pd_s):
    ele1 = geo_s[0]
    geo_s[0] = pd.NA
    ret1 = geo_s.fillna(ele1)

    ele2 = pd_s[0]
    pd_s[0] = pd.NA
    ret2 = pd_s.fillna(ele2)
    pd.testing.assert_series_equal(ret1, ret2, check_dtype=False)


def test_interpolate(geo_s, pd_s):
    geo_s[1] = pd.NA
    geo_s[2] = pd.NA
    ret1 = geo_s.interpolate(method='pad', limit=2)

    pd_s[1] = pd.NA
    pd_s[2] = pd.NA
    ret2 = pd_s.interpolate(method='pad', limit=2)

    pd.testing.assert_series_equal(ret1, ret2, check_dtype=False)


@pytest.mark.skip("not support skew")
def test_skew():
    pass


@pytest.mark.skip("not support std")
def test_std():
    pass


@pytest.mark.skip("not support var")
def test_var():
    pass


@pytest.mark.skip("not support sum")
def test_sum():
    pass


@pytest.mark.skip("not support kurtosis")
def test_kurtosis():
    pass


def test_unique(geo_s, pd_s):
    ret1 = geo_s.unique()
    ret2 = pd_s.unique()

    assert all(ret1 == ret2)

def test_nunique(geo_s, pd_s):
    ret1 = geo_s.nunique()
    ret2 = pd_s.nunique()
    assert ret1 == ret2

def test_isunique(geo_s, pd_s):
    ret1 = geo_s.is_unique
    ret2 = pd_s.is_unique
    assert ret1 == ret2


def test_is_monotonic(geo_s, pd_s):
    ret1 = geo_s.is_monotonic
    ret2 = pd_s.is_monotonic
    assert ret1 == ret2


def test_is_monotonic_increasing(geo_s, pd_s):
    ret1 = geo_s.is_monotonic_increasing
    ret2 = pd_s.is_monotonic_increasing
    assert ret1 == ret2


def test_is_monotonic_decreasing(geo_s, pd_s):
    ret1 = geo_s.is_monotonic_decreasing
    ret2 = pd_s.is_monotonic_decreasing
    assert ret1 == ret2


@pytest.mark.skip("to do")
def test_value_counts():
    pass


def test_align(geo_s, pd_s):
    other = pd.Series([1, 2, 3, 4])
    ops = ["left", "right", "outer"]
    for op in ops:
        g_1, _ = geo_s.align(other, join=op)
        p_1, _ = pd_s.align(other, join=op)
        pd.testing.assert_series_equal(g_1, p_1, check_dtype=False)


def test_drop(geo_s, pd_s):
    indexs = ['A', 'B', 'C', 'D', 'E', 'F', "G", "H", "I"]
    geo_s.index = indexs
    pd_s.index = indexs
    ret1 = geo_s.drop(labels=["A", "B"])
    ret2 = pd_s.drop(labels=["A", "B"])
    pd.testing.assert_series_equal(ret1, ret2, check_dtype=False)


def test_drop_level(geo_s, pd_s):
    indexs = pd.MultiIndex.from_product([['one', 'two', 'three'], ['a', 'b', 'c']])
    geo_s.index = indexs
    pd_s.index = indexs
    ret1 = geo_s.droplevel(0)
    ret2 = pd_s.droplevel(0)
    pd.testing.assert_series_equal(ret1, ret2, check_dtype=False)


# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.view.html
@pytest.mark.skip("not support view")
def test_view():
    pass


# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.append.html
def test_append(geo_s, pd_s):
    geo_append_s = geo_s.append(GeoSeries(geo_pickup.to_list()))
    pd_wkb_s2 = trans2wkb4series(pd.Series(geo_pickup.to_list()))
    pd_append_wkb_s = pd_s.append(pd_wkb_s2)
    pd.testing.assert_series_equal(geo_append_s, pd_append_wkb_s, check_dtype=False)


# pd.set_option("max_colwidth", 1000)

# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.replace.html
def test_replace(geo_s, pd_s):
    elem = geo_s[0]
    target = GeoSeries(["POINT(1 1)"])[0]
    geo_replace_s = geo_s.replace(
        elem,
        target)
    pd_replace_wkb_s = pd_s.replace(
        elem,
        target)

    pd.testing.assert_series_equal(geo_replace_s, pd_replace_wkb_s, check_dtype=False)


# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.update.html
def test_update(geo_s, pd_s):
    geo_s.update(GeoSeries(geo_pickup.to_list()))
    pd_pick_s = trans2wkb4series(pd.Series(geo_pickup.to_list()))
    pd_s.update(pd_pick_s)
    pd.testing.assert_series_equal(pd_s, pd_pick_s, check_dtype=False)


# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.asfreq.html
def test_asfreq(geo_s, pd_s):
    geo_asfreq_s = geo_s.asfreq(freq='30s')
    pd_asfreq_s = pd_s.asfreq(freq='30s')
    pd.testing.assert_series_equal(geo_asfreq_s, pd_asfreq_s, check_dtype=False)


# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.asof.html
def test_asof(geo_s, pd_s):
    geo_asof_s = geo_s.asof(20)
    pd_asof_s = pd_s.asof(20)
    assert geo_asof_s == pd_asof_s


# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.shift.html
def test_shift(geo_s, pd_s):
    geo_shift_s = geo_s.shift(periods=3)
    pd_shift_s = pd_s.shift(periods=3)
    pd.testing.assert_series_equal(geo_shift_s, pd_shift_s, check_dtype=False)


# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.first_valid_index.html
def test_first_valid_index(geo_s, pd_s):
    geo_first_valid_index_s = geo_s.first_valid_index()
    pd_first_valid_index_s = pd_s.first_valid_index()
    assert geo_first_valid_index_s == pd_first_valid_index_s


# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.last_valid_index.html
def test_last_valid_index(geo_s, pd_s):
    geo_last_valid_index_s = geo_s.last_valid_index()
    pd_last_valid_index_s = pd_s.last_valid_index()
    assert geo_last_valid_index_s == pd_last_valid_index_s


# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.resample.html
def test_resample(geo_s, pd_s):
    index = pd.date_range('1/1/2000', periods=9, freq='T', tz='US/Central')
    geo_s.index = index
    pd_s.index = index
    geo_resample_s = geo_s.resample('1T').asfreq()
    pd_resamples_s = pd_s.resample('1T').asfreq()
    pd.testing.assert_series_equal(geo_resample_s, pd_resamples_s, check_dtype=False)


# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.tz_convert.html
def test_tz_convert(geo_s, pd_s):
    index = pd.date_range('1/1/2000', periods=9, freq='T', tz='US/Central')
    geo_s.index = index
    pd_s.index = index

    geo_tz_convert_s = geo_s.tz_convert(tz='Europe/Berlin')
    pd_tz_convert_s = pd_s.tz_convert(tz='Europe/Berlin')
    pd.testing.assert_series_equal(geo_tz_convert_s, pd_tz_convert_s, check_dtype=False)


# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.tz_localize.html
def test_tz_localize(geo_s, pd_s):
    count = geo_s.count()
    index = pd.date_range('2018-10-28', periods=count, freq='H')
    geo_s.index = index
    pd_s.index = index

    geo_tz_localize_s = geo_s.tz_localize('CET', ambiguous='NaT')
    pd_tz_localize_s = pd_s.tz_localize('CET', ambiguous='NaT')
    pd.testing.assert_series_equal(geo_tz_localize_s, pd_tz_localize_s, check_dtype=False)

    geo_tz_localize_s2 = geo_s.tz_localize('UTC', ambiguous='infer')
    pd_tz_localize_s2 = pd_s.tz_localize('UTC', ambiguous='infer')
    pd.testing.assert_series_equal(geo_tz_localize_s2, pd_tz_localize_s2, check_dtype=False)


# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.at_time.html
def test_at_time(geo_s, pd_s):
    count = geo_s.count()
    index = pd.date_range('2018-10-28', periods=count, freq='H')
    geo_s.index = index
    pd_s.index = index

    geo_at_time_s = geo_s.at_time('2018-10-28')
    pd_at_time_s = pd_s.at_time('2018-10-28')
    pd.testing.assert_series_equal(geo_at_time_s, pd_at_time_s, check_dtype=False)


# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.between_time.html
def test_between_time(geo_s, pd_s):
    count = geo_s.count()
    index = pd.date_range('2018-10-28', periods=count, freq='H')
    geo_s.index = index
    pd_s.index = index

    geo_between_time_s = geo_s.between_time('1:00', '4:45')
    pd_betweem_time_s = pd_s.between_time('1:00', '4:45')
    pd.testing.assert_series_equal(geo_between_time_s, pd_betweem_time_s, check_dtype=False)


# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.tshift.html
def test_tshift():
    pass


# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.slice_shift.html
def test_slice_shift(geo_s, pd_s):
    geo_slice_shift_s = geo_s.slice_shift()
    pd_slice_shift_s = pd_s.slice_shift()
    pd.testing.assert_series_equal(geo_slice_shift_s, pd_slice_shift_s, check_dtype=False)


# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.argsort.html
def test_argsort(geo_s, pd_s):
    geo_argsort_s = geo_s.argsort()
    pd_argsort_s = pd_s.argsort()
    pd.testing.assert_series_equal(geo_argsort_s, pd_argsort_s, check_dtype=False)


# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.argmin.html
@pytest.mark.skip("not support argmin")
def test_argmin():
    pass


# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.argmax.html
@pytest.mark.skip("not support argmax")
def test_argmax():
    pass


# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.reorder_levels.html
def test_reorder_levels():
    pass


# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.sort_values.html
def test_sort_values(geo_s, pd_s):
    geo_sort_values = geo_s.sort_values(ascending=True)
    pd_sort_values = pd_s.sort_values(ascending=True)
    pd.testing.assert_series_equal(geo_sort_values, pd_sort_values, check_dtype=False)


# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.sort_index.html
def test_sort_index(geo_s, pd_s):
    geo_sort_index = geo_s.sort_index(ascending=False)
    pd_sort_index = pd_s.sort_index(ascending=False)
    pd.testing.assert_series_equal(geo_sort_index, pd_sort_index, check_dtype=False)


# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.swaplevel.html
def test_swaplevel(geo_s, pd_s):
    index = pd.MultiIndex.from_product([['one', 'two', 'three'], ['a', 'b', 'c']])
    geo_s.index = index
    pd_s.index = index

    geo_swaplevel = geo_s.swaplevel()
    pd_swaplevel = pd_s.swaplevel()
    pd.testing.assert_series_equal(geo_swaplevel, pd_swaplevel, check_dtype=False)


# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.unstack.html
def test_unstack(geo_s, pd_s):
    index = pd.MultiIndex.from_product([['one', 'two', 'three'], ['a', 'b', 'c']])
    geo_s.index = index
    pd_s.index = index

    geo_unstack_df = geo_s.unstack(level=-1)
    pd_unstack_df = pd_s.unstack(level=-1)
    pd.testing.assert_frame_equal(geo_unstack_df, pd_unstack_df, check_dtype=False)


# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.explode.html
def test_explode(geo_s, pd_s):
    geo_explode_s = geo_s.explode()
    pd_explode_s = pd_s.explode()
    pd.testing.assert_series_equal(geo_explode_s, pd_explode_s, check_dtype=False)


# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.searchsorted.html
def test_searchsorted(geo_s, pd_s):
    s1 = "POINT(1 1)"
    geo_searchsorted = geo_s.searchsorted(GeoSeries(s1))
    pd_searchsorted = pd_s.searchsorted(trans2wkb4series(pd.Series(s1)))
    assert geo_searchsorted == pd_searchsorted


# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.ravel.html
def test_ravel(geo_s, pd_s):
    geo_ravel = geo_s.ravel()
    pd_ravel = pd_s.ravel()
    assert (geo_ravel == pd_ravel).all()


# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.repeat.html
def test_repeat(geo_s, pd_s):
    geo_repeat = geo_s.repeat(2)
    pd_repeat = pd_s.repeat(2)
    pd.testing.assert_series_equal(geo_repeat, pd_repeat, check_dtype=False)


# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.squeeze.html
def test_squeeze(geo_s, pd_s):
    geo_squeeze = geo_s.squeeze()
    pd_squeeze = pd_s.squeeze()
    pd.testing.assert_series_equal(geo_squeeze, pd_squeeze, check_dtype=False)


def test_eq(geo_s, pd_s):
    r = geo_s.eq(pd_s[0])
    assert r[0]
    assert not r[1:].any()

    r = geo_s.eq(pd_s)
    assert r.all()


@pytest.mark.skip("not support product")
def test_product():
    pass


# def test_apply(geo_s, pd_s):
#     # bug self_constructor
#     geo_r = geo_s.apply(lambda x: x)
#     pd_r = pd_s.apply(lambda x: x)
#     pd.testing.assert_series_equal(geo_r, pd_r, check_dtype=False)
#
#     def f(s):
#         return arctern.ST_AsText(s)[0]
#
#     geo_r = geo_s.apply(f)
#     pd_r = pd_s.apply(f)
#     pd.testing.assert_series_equal(geo_r, pd_r, check_dtype=False)


def test_agg(geo_s, pd_s):
    def agg(s):
        return arctern.ST_Union_Aggr(s)[0]

    geo_r = geo_s.agg(agg)
    pd_r = pd_s.agg(agg)
    pd.testing.assert_series_equal(geo_r, pd_r, check_dtype=False)


def test_aggrate(geo_s, pd_s):
    def agg(s):
        return arctern.ST_Union_Aggr(s)[0]

    geo_r = geo_s.agg(agg)
    pd_r = pd_s.agg(agg)
    pd.testing.assert_series_equal(geo_r, pd_r, check_dtype=False)


# def test_transform(geo_s, pd_s):
#     # transform bug self_constructor
#     def f(s):
#         return arctern.ST_AsText(s)[0]
#
#     geo_r = geo_s.transform(f)
#     pd_r = pd_s.transform(f)
#     pd.testing.assert_series_equal(geo_r, pd_r, check_dtype=False)

# def test_map(geo_s, pd_s):
#     #bug self_constructor
#   map_dict = {pd_s[0]: 'POINT (1 1)'}
#   geo_r = geo_s.map(map_dict).dropna()
#   pd_r = pd_s.map(map_dict).dropna()
#   pd.testing.assert_series_equal(geo_r, pd_r)

def test_group_by(geo_s, pd_s):
    def agg(s):
        return arctern.ST_Union_Aggr(s)[0]

    group = [0, 1, 0] * 3
    geo_r = geo_s.groupby(group).agg(agg)
    pd_r = pd_s.groupby(group).agg(agg)
    pd.testing.assert_series_equal(geo_r, pd_r, check_dtype=False)

    geo_r = geo_s.groupby(level=0).agg(agg)
    pd_r = pd_s.groupby(level=0).agg(agg)
    pd.testing.assert_series_equal(geo_r, pd_r, check_dtype=False)

    geo_r = geo_s.groupby(geo_s == geo_s[0]).agg(agg)
    pd_r = pd_s.groupby(pd_s == pd_s[0]).agg(agg)
    pd.testing.assert_series_equal(geo_r, pd_r, check_dtype=False)


def test_rolling(geo_s, pd_s):
    with pytest.raises(Exception):
        pd_s.rolling(2).sum()

    with pytest.raises(Exception):
        geo_s.rolling(2).sum()


# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.expanding.html#pandas-series-expanding
@pytest.mark.skip("not support expanding")
def test_expanding():
    pass


# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.ewm.html
@pytest.mark.skip("not support ewm")
def test_ewm():
    pass


def test_pipe(geo_s, pd_s):
    def f(s):
        return arctern.ST_AsText(s)

    geo_r = geo_s.pipe(f)
    pd_r = pd_s.pipe(f)
    pd.testing.assert_series_equal(geo_r, pd_r)


# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.abs.html
@pytest.mark.skip("not support abs")
def test_abs():
    pass


# @pytest.mark.skip("not support all")
# def test_all(geo_s, pd_s):
#     print(pd_s.all())
#     geo_s.all()


# @pytest.mark.skip("not support any")
# def test_any(geo_s, pd_s):
#     pd_s.any()
#     geo_s.any()


@pytest.mark.skip("not support autocorr")
def test_autocorrr():
    pass


def test_between(geo_s, pd_s):
    with pytest.raises(TypeError):
        geo_s.between(pd_s[0], pd_s[2])
        # with pytest.raises(TypeError):
    pd_s.between(pd_s[0], pd_s[2])


# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.clip.html
@pytest.mark.skip("not support clip")
def test_clip():
    pass


# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.corr.html
@pytest.mark.skip("not support corr")
def test_corr():
    pass


def test_count(geo_s, pd_s):
    assert geo_s.count() == pd_s.count()


# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.cov.html
@pytest.mark.skip("not support cov")
def test_cov():
    pass


# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.cummax.html
@pytest.mark.skip("not support cummax")
def test_cummax():
    pass


@pytest.mark.skip("not support cummin")
def test_cummin():
    pass


@pytest.mark.skip("not support cumprod")
def test_cumprod():
    pass


@pytest.mark.skip("not support cumsum")
def test_cumsum():
    pass


# def test_describe(geo_s, pd_s):
#     pd_s.describe()
#     geo_s.describe()

@pytest.mark.skip("not support diff")
def test_diff():
    pass


def test_factorize(geo_s, pd_s):
    geo_codes, geo_uniques = pd.factorize(geo_s)
    pd_codes, pd_uniques = pd.factorize(pd_s)

    assert (geo_codes == pd_codes).all()
    assert (geo_uniques == pd_uniques).all()


@pytest.mark.skip("not support kurt")
def test_kurt():
    pass


@pytest.mark.skip("not support mad")
def test_mad():
    pass


@pytest.mark.skip("not support max")
def test_max():
    pass


@pytest.mark.skip("not support mean")
def test_mean():
    pass


@pytest.mark.skip("not support median")
def test_medina():
    pass


@pytest.mark.skip("not support min")
def test_min():
    pass


@pytest.mark.skip("not support mode")
def test_mode():
    pass


@pytest.mark.skip("not support nlargest")
def test_nlargest(geo_s):
    with pytest.raises(TypeError):
        geo_s.nlargest()


@pytest.mark.skip("not support nsmallest")
def test_nsmallest(geo_s):
    with pytest.raises(TypeError):
        geo_s.nsmallest()


@pytest.mark.skip("not support pct_change")
def test_pct_change():
    pass


@pytest.mark.skip("not support prod")
def test_prod():
    pass


@pytest.mark.skip("not support quantile")
def test_quantile():
    pass


@pytest.mark.skip("not support rank")
def test_rank():
    pass


@pytest.mark.skip("not support sem")
def test_sem():
    pass


def test_drop_duplicates(geo_s, pd_s):
    geo_s[geo_s.count()] = geo_s[0]
    pd_s[pd_s.count()] = pd_s[0]
    pd.testing.assert_series_equal(geo_s, pd_s, check_dtype=False)

    geo_r = geo_s.drop_duplicates()
    pd_r = pd_s.drop_duplicates()
    pd.testing.assert_series_equal(geo_r, pd_r, check_dtype=False)

    geo_r = geo_s.drop_duplicates(keep='last')
    pd_r = pd_s.drop_duplicates(keep='last')
    pd.testing.assert_series_equal(geo_r, pd_r, check_dtype=False)


def test_duplicated(geo_s, pd_s):
    geo_s[geo_s.count()] = geo_s[0]
    pd_s[pd_s.count()] = pd_s[0]
    pd.testing.assert_series_equal(geo_s, pd_s, check_dtype=False)

    geo_r = geo_s.duplicated()
    pd_r = pd_s.duplicated()
    pd.testing.assert_series_equal(geo_r, pd_r)

    geo_r = geo_s.duplicated(keep='last')
    pd_r = pd_s.duplicated(keep='last')
    pd.testing.assert_series_equal(geo_r, pd_r)


# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.html
def test_Series(geo_s, pd_s):
    pd.testing.assert_series_equal(geo_s, pd_s, check_dtype=False)  # (as expected)


# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.index.html
def test_index(geo_s, pd_s):
    geo_res = geo_s.index
    pd_res = pd_s.index
    assert (geo_res == pd_res).all()


# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.array.html
def test_array(geo_s, pd_s):
    geo_res = geo_s.array
    pd_res = pd_s.array
    assert (geo_res == pd_res).all()


# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.values.html
def test_values(geo_s, pd_s):
    geo_res = geo_s.values
    pd_res = pd_s.values
    assert (geo_res == pd_res).all()


# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.dtype.html
def test_dtype(geo_s):
    assert geo_s.dtype == 'GeoDtype'


# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.shape.html
def test_shape(geo_s, pd_s):
    geo_res = geo_s.shape
    pd_res = pd_s.shape
    assert geo_res == pd_res


# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.nbytes.html
def test_nbytes(geo_s, pd_s):
    geo_res = geo_s.nbytes
    pd_res = pd_s.nbytes
    assert geo_res == pd_res


# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.ndim.html
def test_ndim(geo_s, pd_s):
    geo_res = geo_s.ndim
    pd_res = pd_s.ndim
    assert geo_res == pd_res


# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.size.html
def test_size(geo_s, pd_s):
    geo_res = geo_s.size
    pd_res = pd_s.size
    assert geo_res == pd_res


# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.T.html
def test_T(geo_s, pd_s):
    geo_res = geo_s.T
    pd_res = pd_s.T
    pd.testing.assert_series_equal(geo_res, pd_res, check_dtype=False)  # (as expected)

# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.memory_usage.html
# def test_memory_usage(geo_s, pd_s):
#     geo_res = geo_s.memory_usage()
#     pd_res = pd_s.memory_usage()
#     assert geo_res == pd_res


# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.hasnans.html
def test_hasnans(geo_s, pd_s):
    geo_res = geo_s.hasnans
    pd_res = pd_s.hasnans
    assert geo_res == pd_res


# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.empty.html
def test_empty(geo_s, pd_s):
    geo_res = geo_s.empty
    pd_res = pd_s.empty
    assert geo_res == pd_res


# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.dtypes.html
def test_dtypes(geo_s):
    # geo_s = GeoSeries(geo_dropoff.to_list())
    assert geo_s.dtypes == 'GeoDtype'


# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.name.html
def test_name(geo_s, pd_s):
    geo_res = geo_s.name
    pd_res = pd_s.name
    assert geo_res == pd_res


# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.astype.html
def test_astype(geo_s, pd_s):
    geo_res = geo_s.astype('object')
    pd_res = pd_s.astype('object')
    pd.testing.assert_series_equal(geo_res, pd_res, check_dtype=False)  # (as expected)


# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.convert_dtypes.html
def test_convert_dtypes(geo_s, pd_s):
    geo_res = geo_s.convert_dtypes()
    pd_res = pd_s.convert_dtypes()
    pd.testing.assert_series_equal(geo_res, pd_res, check_dtype=False)  # (as expected)


# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.infer_objects.html
def test_infer_objects(geo_s, pd_s):
    geo_res = geo_s.infer_objects()
    pd_res = pd_s.infer_objects()
    pd.testing.assert_series_equal(geo_res, pd_res, check_dtype=False)  # (as expected)


# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.copy.html
def test_copy(geo_s, pd_s):
    geo_res = geo_s.copy()
    pd_res = pd_s.copy()
    pd.testing.assert_series_equal(geo_res, pd_res, check_dtype=False)  # (as expected)


# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.bool.html
# Only support boolean
# def test_bool(geo_s, pd_s):
#     geo_s = GeoSeries(geo_dropoff.to_list())
#     pd_s = pd.Series(geo_dropoff.to_list())
#     pd_s = trans2wkb4series(pd_s)
#     pd.testing.assert_series_equal(geo_s,pd_s,check_dtype=False) # (as expected)
# #     geo_res = geo_s.astype(object).all().bool()
#     pd_res = pd_s.all().bool()

# #     assert geo_res == pd_res

# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.to_numpy.html
def test_to_numpy(geo_s, pd_s):
    geo_res = geo_s.to_numpy()
    pd_res = pd_s.to_numpy()
    assert (geo_res == pd_res).all()


# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.to_period.html
def test_to_period(geo_s, pd_s):
    index = pd.date_range('1/1/2000', periods=geo_dropoff.count(), freq='T')
    geo_s.index = index
    pd_s.index = index

    geo_res = geo_s.to_period(copy=False, freq='30T')
    pd_res = pd_s.to_period(copy=False, freq='30T')
    pd.testing.assert_series_equal(geo_res, pd_res, check_dtype=False)


# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.to_timestamp.html
def test_to_timestamp(geo_s, pd_s):
    index = pd.period_range('1/1/2000', periods=geo_dropoff.count(), freq='S')
    geo_s.index = index
    pd_s.index = index

    geo_res = geo_s.to_timestamp()
    pd_res = pd_s.to_timestamp()
    pd.testing.assert_series_equal(geo_res, pd_res, check_dtype=False)


# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.to_list.html
def test_to_list(geo_s, pd_s):
    geo_res = geo_s.to_list()
    pd_res = pd_s.to_list()
    assert geo_res == pd_res


# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.__array__.html
def test___array__(geo_s, pd_s):
    geo_res = geo_s.__array__()
    pd_res = pd_s.__array__()
    assert (geo_res == pd_res).all()


# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.get.html
def test_get(geo_s, pd_s):
    geo_res = geo_s.get(1)
    pd_res = pd_s.get(1)
    assert geo_res == pd_res


# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.at.html
def test_at(geo_s, pd_s):
    geo_res = geo_s.at[2]
    pd_res = pd_s.at[2]
    assert geo_res == pd_res


# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.iat.html
def test_iat(geo_s, pd_s):
    geo_res = geo_s.iat[2]
    pd_res = pd_s.iat[2]
    assert geo_res == pd_res


# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.iloc.html
def test_iloc(geo_s, pd_s):
    geo_res = geo_s.iloc[:2]
    pd_res = pd_s.iloc[:2]
    pd.testing.assert_series_equal(geo_res, pd_res, check_dtype=False)


# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.loc.html
def test_loc(geo_s, pd_s):
    geo_res = geo_s.loc[:2]
    pd_res = pd_s.loc[:2]
    pd.testing.assert_series_equal(geo_res, pd_res, check_dtype=False)  # (as expected)


# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.__iter__.html
# ISSUE : https://github.com/arctern-io/arctern/issues/694
def test_iter(geo_s, pd_s):
    geo_res = []
    pd_res = []
    for i in geo_s.items():
        geo_res.append(i)
    for j in pd_s.items():
        pd_res.append(j)

    assert geo_res == pd_res


# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.items.html
def test_items(geo_s, pd_s):
    geo_res = []
    pd_res = []
    for i in geo_s.items():
        geo_res.append(i)
    for j in pd_s.items():
        pd_res.append(j)

    assert geo_res == pd_res


# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.iteritems.html
def test_iteritems(geo_s, pd_s):
    geo_res = []
    pd_res = []
    for i in geo_s.iteritems():
        geo_res.append(i)
    for j in pd_s.iteritems():
        pd_res.append(j)

    assert geo_res == pd_res


# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.keys.html
def test_keys(geo_s, pd_s):
    geo_res = geo_s.keys()
    pd_res = pd_s.keys()
    assert (geo_res == pd_res).all()


# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.pop.html
def test_pop(geo_s, pd_s):
    geo_res = geo_s.pop(1)
    pd_res = pd_s.pop(1)
    assert geo_res == pd_res


# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.item.html
def test_item(geo_s, pd_s):
    geo_res = geo_s.head(1).item()
    pd_res = pd_s.head(1).item()
    assert geo_res == pd_res


# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.xs.html
# GeoSeries not supported
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.add.html
# GeoSeries not supported
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.sub.html
# GeoSeries not supported
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.mul.html
# GeoSeries not supported
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.div.html
# GeoSeries not supported
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.truediv.html
# GeoSeries not supported
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.floordiv.html
# GeoSeries not supported
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.mod.html
# GeoSeries not supported
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.pow.html
# GeoSeries not supported
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.radd.html
# GeoSeries not supported
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.rsub.html
# GeoSeries not supported
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.rmul.html
# GeoSeries not supported
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.rdiv.html
# GeoSeries not supported
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.rtruediv.html
# GeoSeries not supported
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.rfloordiv.html
# GeoSeries not supported
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.rmod.html
# GeoSeries not supported
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.rpow.html
# GeoSeries not supported
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.combine.html
def test_combine(geo_s, pd_s):
    geo_s_combine = GeoSeries(geo_pickup.to_list())
    pd_s_combine = pd.Series(geo_pickup.to_list())
    pd_s_combine_wkb = trans2wkb4series(pd_s_combine)
    pd.testing.assert_series_equal(geo_s, pd_s, check_dtype=False)  # (as expected)
    pd.testing.assert_series_equal(geo_s_combine, pd_s_combine_wkb, check_dtype=False)  # (as expected)

    take_any = lambda p1, p2: p2 if (p1 != p2) else p1
    geo_res = geo_s.combine(geo_s_combine, take_any)
    pd_res = pd_s.combine(pd_s_combine_wkb, take_any)
    pd.testing.assert_series_equal(geo_res, pd_res, check_dtype=False)  # (as expected)


# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.combine_first.html
def test_combine_first(geo_s, pd_s):
    geo_s_combine = GeoSeries(geo_pickup.to_list())
    pd_s_combine = pd.Series(geo_pickup.to_list())
    pd_s_combine_wkb = trans2wkb4series(pd_s_combine)
    pd.testing.assert_series_equal(geo_s, pd_s, check_dtype=False)  # (as expected)
    pd.testing.assert_series_equal(geo_s_combine, pd_s_combine_wkb, check_dtype=False)  # (as expected)

    geo_res = geo_s.combine_first(geo_s_combine)
    pd_res = pd_s.combine_first(pd_s_combine_wkb)
    pd.testing.assert_series_equal(geo_res, pd_res, check_dtype=False)  # (as expected)


# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.round.html
# GeoSeries not supported
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.lt.html
# GeoSeries not supported
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.gt.html
# GeoSeries not supported
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.le.html
# GeoSeries not supported
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.ge.html
# GeoSeries not supported
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.ne.html

def test_ne(geo_s, pd_s):
    geo_s_compare = GeoSeries(geo_pickup.to_list())
    pd_s_compare = pd.Series(geo_pickup.to_list())
    pd_s_compare_wkb = trans2wkb4series(pd_s_compare)
    pd.testing.assert_series_equal(geo_s, pd_s, check_dtype=False)  # (as expected)
    pd.testing.assert_series_equal(geo_s_compare, pd_s_compare_wkb, check_dtype=False)  # (as expected)

    geo_res = geo_s.ne(geo_s_compare)
    pd_res = pd_s.ne(pd_s_compare_wkb)
    assert (geo_res == pd_res).all()

def test_to_string(geo_s):
    pd_s = pd.Series(geo_dropoff.to_list())
    geo_s_string = geo_s.to_string()
    pd_s_string = pd_s.to_string()
    assert geo_s_string == pd_s_string
