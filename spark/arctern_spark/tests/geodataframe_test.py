import pandas as pd
import pytest
from databricks.koalas import Series
from arctern_spark.geodataframe import GeoDataFrame
from arctern_spark.geoseries import GeoSeries

wkt = "point (1 1)"
wkb = b'\x00\x00\x00\x00\x01?\xf0\x00\x00\x00\x00\x00\x00?\xf0\x00\x00\x00\x00\x00\x00'


class TestConstructor:
    @pytest.mark.parametrize("data", [
        wkt,
        wkb,
        [wkt],
        [wkb],
        {0: wkt},
        {0: wkb},
        pd.Series(wkt),
        pd.Series(wkb),
    ])
    def test_from_pandas_data(self, data):
        s = GeoSeries(data)
        assert s.to_wkt().to_list() == ["POINT (1 1)"]

    @pytest.mark.parametrize("data", [
        Series(wkt),
        Series(wkb),
    ])
    def test_from_koalas(self, data):
        s = GeoSeries(data)
        assert s.to_wkt().to_list() == ["POINT (1 1)"]


# test operation will not lose crs info
class TestCRS:
    def test_init_from_geoseries(self):
        crs = "EPSG:4326"
        gs = GeoSeries("point (1 2)", name='a', crs=crs)
        gdf = GeoDataFrame(gs)
        assert gdf['a'].crs == crs

    def test_implicitly_set_geometries(self):
        crs = "EPSG:4326"
        psb = pd.Series("point (99 99)", name='b')
        psa = pd.Series("point (1 2)", name='a')
        gdf = GeoDataFrame({"a": psa, "b": psb}, geometries=['a'], crs=crs)
        assert gdf['a'].crs == crs

    def test_explicitly_set_geometries(self):
        psb = pd.Series("point (99 99)", name='b')
        psa = pd.Series("point (1 2)", name='a')
        gdf = GeoDataFrame({"a": psa, "b": psb}, geometries=['a'], crs="EPSG:4326")
        gdf.set_geometry('b', "EPSG:3857")
        assert gdf['a'].crs == "EPSG:4326"
        assert gdf['b'].crs == "EPSG:3857"

    def test_setitem_getitem(self):
        # set or get item with scalar key
        gdf = GeoDataFrame([1], columns=['seq'])
        gdf['a'] = GeoSeries("point (1 2)", crs="EPSG:4326")
        gdf['b'] = GeoSeries("point (99 99)")
        assert gdf['a'].crs == "EPSG:4326"
        assert gdf['b'].crs is None

        # set or get item with slice key
        gdf1 = GeoDataFrame([1], columns=['seq'])
        gdf1[['a', 'b']] = gdf[['a', 'b']]
        r = gdf1[:]
        assert r['a'].crs == "EPSG:4326"
        assert r['b'].crs is None

    def test_geoseries_modify_crs(self):
        gdf = GeoDataFrame(GeoSeries("point (1 2)", name='a', crs=None))
        assert gdf['a'].crs is None

        # modify geoseries crs
        gdf['a'].crs = "EPSG:4326"
        assert gdf['a'].crs == "EPSG:4326"
