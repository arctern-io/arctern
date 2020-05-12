from arcternpandas.geoseries import GeoSeries
from arcternpandas.geoarray import from_wkt, from_wkb
from pandas import Series

if __name__ == "__main__":
    d = from_wkt(["Point (1 2)", "Point (2 3)", "LINESTRING (0 0, 0 1, 0 0)"])
    # s = GeoSeries(Series([1, 2, 3]))
    s = GeoSeries(d, index=['a', 'b', 'c'], name="hahh")
    print(s)

    print("-" * 50)
    s = GeoSeries([d[0], d[1]])
    print(s)

    print("-" * 50)
    s = GeoSeries(["point (1 2)", None, "point (1 3)"])
    print(s)

    print("-" * 50)
    print(s.isna)
    print("-" * 50)
    print(s.is_valid)
    print("-" * 50)
    print(s.area)
    print("-" * 50)
    s.set_crs("EPSG:4326")
    print(s.to_crs("EPSG:3857"))

    print("-" * 50)
    s1 = GeoSeries({'a': "point (1 2)"})
    print(s1)

    print("-" * 50)
    print(s.intersects(s1))
