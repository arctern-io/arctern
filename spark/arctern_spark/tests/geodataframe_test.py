import numpy as np
import pandas as pd
import pytest
from arctern_spark.geodataframe import GeoDataFrame
from arctern_spark.geoseries import GeoSeries
from databricks.koalas import Series

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


class TestOp:
    def test_merge(self):
        data1 = {
            "A": range(5),
            "B": np.arange(5.0),
            "other_geom": range(5),
            "geometry": ["POINT (0 0)", "POINT (1 1)", "POINT (2 2)", "POINT (3 3)", "POINT (4 4)"],
        }
        gdf1 = GeoDataFrame(data1, geometries=["geometry"], crs=["epsg:4326"])
        data2 = {
            "A": range(5),
            "location": ["POINT (3 0)", "POINT (1 6)", "POINT (2 4)", "POINT (3 4)", "POINT (4 2)"],
        }
        gdf2 = GeoDataFrame(data2, geometries=["location"], crs=["epsg:4326"])
        result = gdf1.merge(gdf2, left_on="A", right_on="A")
        assert isinstance(result, GeoDataFrame)
        assert isinstance(result["geometry"], GeoSeries)
        assert result.location.crs == "EPSG:4326"

    def test_merge_same_column_name(self):
        data1 = {
            "A": range(5),
            "B": np.arange(5.0),
            "other_geom": range(5),
            "location": ["POINT (0 0)", "POINT (1 1)", "POINT (2 2)", "POINT (3 3)", "POINT (4 4)"],
        }
        gdf1 = GeoDataFrame(data1, geometries=["location"], crs=["epsg:4326"])
        data2 = {
            "A": range(5),
            "location": ["POINT (3 0)", "POINT (1 6)", "POINT (2 4)", "POINT (3 4)", "POINT (4 2)"],
        }
        gdf2 = GeoDataFrame(data2, geometries=["location"], crs=["epsg:3857"])
        result = gdf1.merge(gdf2, left_on="A", right_on="A")
        assert isinstance(result, GeoDataFrame)
        assert isinstance(result["location_x"], GeoSeries)
        assert result.location_x.crs == "EPSG:4326"
        assert result.location_y.crs == "EPSG:3857"

    def test_merge_suffixed_column_name(self):
        data1 = {
            "A": range(5),
            "B": np.arange(5.0),
            "other_geom": range(5),
            "location_x": ["POINT (0 0)", "POINT (1 1)", "POINT (2 2)", "POINT (3 3)", "POINT (4 4)"],
        }
        gdf1 = GeoDataFrame(data1, geometries=["location_x"], crs=["epsg:4326"])
        data2 = {
            "A": range(5),
            "location_y": ["POINT (3 0)", "POINT (1 6)", "POINT (2 4)", "POINT (3 4)", "POINT (4 2)"],
        }
        gdf2 = GeoDataFrame(data2, geometries=["location_y"], crs=["epsg:3857"])
        result = gdf1.merge(gdf2, left_on="A", right_on="A")
        assert isinstance(result, GeoDataFrame)
        assert isinstance(result["location_x"], GeoSeries)
        assert result.location_x.crs == "EPSG:4326"
        assert result.location_y.crs == "EPSG:3857"

    # open this test when thread https://github.com/databricks/koalas/issues/1633 solved
    # def test_dissolve(self):
    #     data = {
    #         "A": range(5),
    #         "B": np.arange(5.0),
    #         "other_geom": [1, 1, 1, 2, 2],
    #         "geo1": ["POINT (0 0)", "POINT (1 1)", "POINT (2 2)", "POINT (3 3)", "POINT (4 4)"],
    #     }
    #     gdf = GeoDataFrame(data, geometries=["geo1"], crs=["epsg:4326"])
    #     dissolve_gdf = gdf.disolve(by="other_geom", col="geo1")
    #     assert dissolve_gdf["geo1"].to_wkt()[1] == "MULTIPOINT (0 0,1 1,2 2)"
    #     assert dissolve_gdf["geo1"].to_wkt()[2] == "MULTIPOINT (3 3,4 4)"
