from arctern_spark.geoseries import GeoSeries

def test_ST_IsValid():
    data = GeoSeries(["POINT (1.3 2.6)", "POINT (2.6 4.7)"])
    rst = data.is_valid
    assert rst[0]
    assert rst[1]

if __name__ == "__main__":
    test_ST_IsValid()

