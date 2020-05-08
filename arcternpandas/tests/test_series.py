from arcternpandas.geoseries import GeoSeries
from arcternpandas.geoarry import from_wkt
import arctern
from pandas import Series

if __name__ == "__main__":
    d = from_wkt(["Point (1 2)", "Point (2 3)", "LINESTRING (0 0, 0 1, 0 0)"])
    # s = GeoSeries(Series([1, 2, 3]))
    s = GeoSeries(d, index=[4, 2, 3])
    # s = GeoSeries(s1)
    print(s)
    print(s.isna)
    print(s.is_valid)
    print(s.area)