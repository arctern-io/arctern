from arcternpandas.geoseries import GeoSeries
from arcternpandas.geoarray import from_wkt
import arctern
from pandas import Series

if __name__ == "__main__":
    d = from_wkt(["Point (1 2)", "Point (2 3)", "LINESTRING (0 0, 0 1, 0 0)"])
    # s = GeoSeries(Series([1, 2, 3]))
    s = GeoSeries(d, index=['a', 'b', 'c'], name="hahh")
    # s = GeoSeries(s1)
    # print(arctern.ST_AsText(s))
    # print(s.isna)
    # print(s.is_valid)
    # print(s.area)
    # s.set_crs("EPSG:4326")
    # print(arctern.ST_AsText(s.to_crs("EPSG:3857")))

    print(arctern.ST_AsText(s))
    print(s.intersects(s))

    print(s[s != s[0]])
