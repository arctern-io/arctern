import pandas as pd
from arctern_spark.geodataframe import GeoDataFrame
from arctern_spark.geoseries import GeoSeries


class TestConstructor:
    pass


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
