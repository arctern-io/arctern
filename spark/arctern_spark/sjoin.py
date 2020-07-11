import warnings

from arctern_spark.geodataframe import GeoDataFrame, GeoSeries


def sjoin(
        left_df, right_df, lcol, rcol, how="inner", op="within", lsuffix="_left", rsuffix="_right"
):
    """
    Spatial join of two GeoDataFrames.

    Parameters
    ----------
    left_df, right_df : GeoDataFrame
    rcol, lcol : str
        Specify geometry columns of left_df, right_df to be joined.
    how : string, default 'inner'
        The type of join:

        # TODO: fix doc
        * 'left': use keys from left_df; retain only left_df geometry column
        * 'right': use keys from right_df; retain only right_df geometry column
        * 'inner':
        * 'full':
    op : string, default 'intersects'
        Binary predicate, one of {'contains', 'within'}.
    lsuffix : string, default 'left'
        Suffix to apply to overlapping column names (left GeoDataFrame).
    rsuffix : string, default 'right'
        Suffix to apply to overlapping column names (right GeoDataFrame).

    Returns
    -------
    GeoDataFrame
        An GeoDataFrame object.

    Examples
    ---------
    >>> import arctern_spark
    >>> from arctern_spark import GeoDataFrame
    >>> points = [
    ...    "Point(1 1)",
    ...     "Point(1 2)",
    ...     "Point(2 1)",
    ...     "Point(2 2)",
    ...     "Point(3 3)",
    ...     "Point(4 5)",
    ...     "Point(8 8)",
    ...     "Point(10 10)",
    ...     ]
    >>>
    >>> polygons = [
    ...     "Polygon((0 0, 3 0, 3.1 3.1, 0 3, 0 0))",
    ...     "Polygon((6 6, 3 6, 2.9 2.9, 6 3, 6 6))",
    ...     "Polygon((6 6, 9 6, 9 9, 6 9, 6 6))",
    ...     "Polygon((100 100, 100 101, 101 101, 101 100, 100 100))",
    ...     ]
    >>> d1 = GeoDataFrame({"A": range(8), "points":  points}, geometries=["points"], crs="EPSG:4326")
    >>> d2 = GeoDataFrame({"A": range(4), "polygons": polygons}, geometries=['polygons'], crs="EPSG:4326")
    >>> rst = arctern_spark.sjoin(d1, d2, "points", "polygons")
    >>> rst
       A_left       points  A_right                                 polygons
    0       4  POINT (3 3)        0  POLYGON ((0 0, 3 0, 3.1 3.1, 0 3, 0 0))
    1       4  POINT (3 3)        1  POLYGON ((6 6, 3 6, 2.9 2.9, 6 3, 6 6))
    2       0  POINT (1 1)        0  POLYGON ((0 0, 3 0, 3.1 3.1, 0 3, 0 0))
    3       5  POINT (4 5)        1  POLYGON ((6 6, 3 6, 2.9 2.9, 6 3, 6 6))
    4       1  POINT (1 2)        0  POLYGON ((0 0, 3 0, 3.1 3.1, 0 3, 0 0))
    5       3  POINT (2 2)        0  POLYGON ((0 0, 3 0, 3.1 3.1, 0 3, 0 0))
    6       6  POINT (8 8)        2      POLYGON ((6 6, 9 6, 9 9, 6 9, 6 6))
    7       2  POINT (2 1)        0  POLYGON ((0 0, 3 0, 3.1 3.1, 0 3, 0 0))
    """
    assert isinstance(left_df, GeoDataFrame), f"'left_df' should be GeoDataFrame, got {type(left_df)}"
    assert isinstance(right_df, GeoDataFrame), f"'right_df' should be GeoDataFrame, got {type(right_df)}"
    allowed_hows = ["left", "right", "inner", "full"]
    assert how in allowed_hows
    if not left_df[lcol].crs == right_df[rcol].crs:
        warnings.warn(("CRS of frames being joined does not match! (%s != %s)" % (left_df.crs, right_df.crs)))

    lsdf = left_df.to_spark()
    rsdf = right_df.to_spark()
    import arctern_spark.scala_wrapper as scala_wrapper
    joined_sdf = getattr(scala_wrapper, "spatial_join")(lsdf, rsdf, lcol, rcol, how, op, lsuffix, rsuffix)
    result = GeoDataFrame(joined_sdf)

    for col in result.columns:
        kser = result[col]
        rcol = col
        if isinstance(kser, GeoSeries):
            pick = left_df
            if col.endswith(lsuffix) and col not in left_df.columns:
                col = col[:-len(lsuffix)]
            elif col.endswith(rsuffix) and col not in right_df.columns:
                col = col[:-len(rsuffix)]
                pick = right_df
            elif col in right_df.columns:
                pick = right_df

            crs = pick._crs_for_cols.get(col, None)
            result.set_geometry(rcol, crs)

    return result
