import warnings
import pandas as pd
import numpy as np


def _clip_points(gdf, poly, col=None):
    from arctern import GeoSeries
    return gdf.iloc[gdf[col].sindex.query(GeoSeries(poly))]


def _clip_line_poly(gdf, poly, col=None):
    from arctern import GeoDataFrame, GeoSeries
    gdf_sub = gdf.iloc[gdf[col].sindex.query(GeoSeries(poly))]

    if isinstance(gdf_sub, GeoDataFrame):
        clipped = gdf_sub.copy()
        clipped[col] = gdf_sub[col].intersection(poly)
    else:
        clipped = gdf_sub.intersection(poly)

    return clipped


def clip(gdf, mask, keep_geom_type=False, col=None):
    """
    Clip points, lines, or polygon geometries to the mask extent.

    Both layers must be in the same Coordinate Reference System (CRS).
    The `gdf` will be clipped to the full extent of the clip object.

    If there are multiple polygons in mask, data from `gdf` will be
    clipped to the total boundary of all polygons in mask.

    Parameters
    ----------
    gdf : GeoDataFrame or GeoSeries
        Vector layer (point, line, polygon) to be clipped to mask.
    mask : GeoSeries or str
        Polygon vector layer used to clip `gdf`.
        The mask's geometry is dissolved into one geometric feature
        and intersected with `gdf`.
    keep_geom_type : boolean, default False
        If True, return only geometries of original type in case of intersection
        resulting in multiple geometry types or GeometryCollections.
        If False, return all resulting geometries (potentially mixed-types).
    col : str
        Specify geometry column.


    Returns
    -------
    GeoDataFrame or GeoSeries
         Vector data (points, lines, polygons) from `gdf` clipped to
         polygon boundary from mask.

    Examples
    --------
    >>> import arctern
    >>> from arctern import GeoDataFrame, GeoSeries
    >>> s3 = GeoSeries(["POLYGON ((2 1,3 1,3 2,2 2,2 1))",
    ...                "POLYGON ((-1 1, 1.5 1, 1.5 2, -1 2, -1 1))",
    ...                "POLYGON ((10 10, 20 10, 20 20, 10 20, 10 10))"])
    >>> d1 = GeoDataFrame({"geo":s3})
    >>> arctern.clip(d1, "POLYGON ((1 1,1 2,2 2,2 1,1 1))", col="geo")
                                               geo
    0                         LINESTRING (2 2,2 1)
    1  POLYGON ((1.5 1.0,1 1,1 2,1.5 2.0,1.5 1.0))
    """
    from arctern import GeoDataFrame, GeoSeries
    if not isinstance(gdf, (GeoDataFrame, GeoSeries)):
        raise TypeError(
            "'gdf' should be GeoDataFrame or GeoSeries, got {}".format(type(gdf))
        )

    if not isinstance(mask, (str, GeoSeries)):
        raise TypeError(
            "'mask' should be GeoDataFrame, got {}".format(type(gdf))
        )

    if isinstance(mask, GeoSeries):
        box_mask = mask.unary_union()
    else:
        mask = GeoSeries(mask)
        box_mask = mask.unary_union()
    box_gdf = gdf[col].unary_union()
    if not box_mask.intersects(box_gdf)[0]:
        return gdf.iloc[:0]
    geom_types = gdf[col].geom_type
    for i, geom_type in enumerate(geom_types):
        geom_types[i] = geom_type[3:]

    poly_idx = np.asarray((geom_types == "POLYGON") | (geom_types == "MULTIPOLYGON"))
    line_idx = np.asarray(
        (geom_types == "LINESTRING")
        | (geom_types == "LINEARRING")
        | (geom_types == "MULTILINESTRING")
    )
    point_idx = np.asarray((geom_types == "POINT") | (geom_types == "MULTIPOINT"))
    geomcoll_idx = np.asarray((geom_types == "GEOMETRYCOLLECTION"))

    if point_idx.any():
        point_gdf = _clip_points(gdf[point_idx], box_mask[0], col)
    else:
        point_gdf = None

    if poly_idx.any():
        poly_gdf = _clip_line_poly(gdf[poly_idx], box_mask[0], col)
    else:
        poly_gdf = None

    if line_idx.any():
        line_gdf = _clip_line_poly(gdf[line_idx], box_mask[0], col)
    else:
        line_gdf = None

    if geomcoll_idx.any():
        geomcoll_gdf = _clip_line_poly(gdf[geomcoll_idx], box_mask[0], col)
    else:
        geomcoll_gdf = None

    order = pd.Series(range(len(gdf)), index=gdf.index)
    concat = pd.concat([point_gdf, line_gdf, poly_gdf, geomcoll_gdf])

    if keep_geom_type:
        geomcoll_concat = (concat[col].geom_type == "GeometryCollection").any()
        geomcoll_orig = geomcoll_idx.any()

        new_collection = geomcoll_concat and not geomcoll_orig
        if geomcoll_orig:
            warnings.warn(
                "keep_geom_type can not be called on a "
                "GeoDataFrame with GeometryCollection."
            )
        else:
            polys = ["POLYGON", "MULTIPOLYGON"]
            lines = ["LINESTRING", "MULTILINESTRING", "LINEARRING"]
            points = ["POINT", "MULTIPOINT"]

            orig_types_total = sum(
                [
                    gdf[col].geom_type.isin(polys).any(),
                    gdf[col].geom_type.isin(lines).any(),
                    gdf[col].geom_type.isin(points).any(),
                ]
            )

            clip_types_total = sum(
                [
                    concat[col].geom_type.isin(polys).any(),
                    concat[col].geom_type.isin(lines).any(),
                    concat[col].geom_type.isin(points).any(),
                ]
            )

            more_types = orig_types_total < clip_types_total
            if orig_types_total > 1:
                warnings.warn(
                    "keep_geom_type can not be called on a mixed type GeoDataFrame."
                )
            elif new_collection or more_types:
                orig_type = gdf[col].geom_type[0][3:]
                if new_collection:
                    concat = concat.explode()
                if orig_type in polys:
                    concat = concat.loc[concat[col].geom_type.isin(polys)]
                elif orig_type in lines:
                    concat = concat.loc[concat[col].geom_type.isin(lines)]

    if len(concat) == 0:
        return gdf.iloc[0]

    if isinstance(concat, GeoDataFrame):
        concat["_order"] = order
        return concat.sort_values(by="_order").drop(columns="_order")

    concat = GeoDataFrame(concat, geometries=[concat.name])
    concat["_order"] = order
    return concat.sort_values(by="_order")[col]
