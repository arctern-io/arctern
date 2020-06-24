package org.apache.spark.sql.arctern

import org.apache.spark.sql.Column
import org.apache.spark.sql.arctern.expressions._
import org.apache.spark.sql.catalyst.expressions.aggregate.AggregateFunction

object functions {
  // Constructor UDF API
  def st_geomfromtext(wkt: Column): Column = Column {
    ST_GeomFromText(Seq(wkt.expr))
  }

  def st_geomfromwkb(wkt: Column): Column = Column {
    ST_GeomFromWKB(Seq(wkt.expr))
  }

  def st_point(x: Column, y: Column): Column = Column {
    ST_Point(Seq(x.expr, y.expr))
  }

  def st_polygonfromenvelope(xMin: Column, yMin: Column, xMax: Column, yMax: Column): Column = Column {
    ST_PolygonFromEnvelope(Seq(xMin.expr, yMin.expr, xMax.expr, yMax.expr))
  }

  def st_geomfromgeojson(json: Column): Column = Column {
    ST_GeomFromGeoJSON(Seq(json.expr))
  }

  def st_astext(geo: Column): Column = Column {
    ST_AsText(Seq(geo.expr))
  }

  def st_asgeojson(geo: Column): Column = Column {
    ST_AsGeoJSON(Seq(geo.expr))
  }

  // Function UDF API
  def st_within(left: Column, right: Column): Column = Column {
    ST_Within(Seq(left.expr, right.expr))
  }

  def st_centroid(geo: Column): Column = Column {
    ST_Centroid(Seq(geo.expr))
  }

  def st_isvalid(geo: Column): Column = Column {
    ST_IsValid(Seq(geo.expr))
  }

  def st_geometrytype(geo: Column): Column = Column {
    ST_GeometryType(Seq(geo.expr))
  }

  def st_issimple(geo: Column): Column = Column {
    ST_IsSimple(Seq(geo.expr))
  }

  def st_npoints(geo: Column): Column = Column {
    ST_NPoints(Seq(geo.expr))
  }

  def st_envelope(geo: Column): Column = Column {
    ST_Envelope(Seq(geo.expr))
  }

  def st_buffer(geo: Column, distance: Column): Column = Column {
    ST_Buffer(Seq(geo.expr, distance.expr))
  }

  def st_precisionreduce(geo: Column, precision: Column): Column = Column {
    ST_PrecisionReduce(Seq(geo.expr, precision.expr))
  }

  def st_intersection(left: Column, right: Column): Column = Column {
    ST_Intersection(Seq(left.expr, right.expr))
  }

  def st_simplifypreservetopology(geo: Column, tolerance: Column): Column = Column {
    ST_SimplifyPreserveTopology(Seq(geo.expr, tolerance.expr))
  }

  def st_convexhull(geo: Column): Column = Column {
    ST_ConvexHull(Seq(geo.expr))
  }

  def st_area(geo: Column): Column = Column {
    ST_Area(Seq(geo.expr))
  }

  def st_length(geo: Column): Column = Column {
    ST_Length(Seq(geo.expr))
  }

  def st_hausdorffdistance(left: Column, right: Column): Column = Column {
    ST_HausdorffDistance(Seq(left.expr, right.expr))
  }

  def st_distance(left: Column, right: Column): Column = Column {
    ST_Distance(Seq(left.expr, right.expr))
  }

  def st_equals(left: Column, right: Column): Column = Column {
    ST_Equals(Seq(left.expr, right.expr))
  }

  def st_touches(left: Column, right: Column): Column = Column {
    ST_Touches(Seq(left.expr, right.expr))
  }

  def st_overlaps(left: Column, right: Column): Column = Column {
    ST_Overlaps(Seq(left.expr, right.expr))
  }

  def st_crosses(left: Column, right: Column): Column = Column {
    ST_Crosses(Seq(left.expr, right.expr))
  }

  def st_contains(left: Column, right: Column): Column = Column {
    ST_Contains(Seq(left.expr, right.expr))
  }

  def st_intersects(left: Column, right: Column): Column = Column {
    ST_Intersects(Seq(left.expr, right.expr))
  }

  def st_distancesphere(left: Column, right: Column): Column = Column {
    ST_DistanceSphere(Seq(left.expr, right.expr))
  }

  def st_transform(geo: Column, sourceCRSCode: Column, targetCRSCode: Column): Column = Column {
    ST_Transform(Seq(geo.expr, sourceCRSCode.expr, targetCRSCode.expr))
  }

  def st_makevalid(geo: Column): Column = Column {
    ST_MakeValid(Seq(geo.expr))
  }

  def st_curvetoline(geo: Column): Column = Column {
    ST_CurveToLine(Seq(geo.expr))
  }

  def st_translate(geo: Column, shifterXValue: Column, shifterYValue: Column): Column = Column {
    ST_Translate(Seq(geo.expr, shifterXValue.expr, shifterYValue.expr))
  }

  def st_rotate(geo: Column, rotationAngle: Column, rotateX: Column, rotateY: Column): Column = Column {
    ST_Rotate(Seq(geo.expr, rotationAngle.expr, rotateX.expr, rotateY.expr))
  }

  def st_symdifference(left: Column, right: Column): Column = Column {
    ST_SymDifference(Seq(left.expr, right.expr))
  }

  def st_difference(left: Column, right: Column): Column = Column {
    ST_Difference(Seq(left.expr, right.expr))
  }

  def st_union(left: Column, right: Column): Column = Column {
    ST_Union(Seq(left.expr, right.expr))
  }

  def st_disjoint(left: Column, right: Column): Column = Column {
    ST_Disjoint(Seq(left.expr, right.expr))
  }

  def st_isempty(geo: Column): Column = Column {
    ST_IsEmpty(Seq(geo.expr))
  }

  def st_boundary(geo: Column): Column = Column {
    ST_Boundary(Seq(geo.expr))
  }

  def st_exteriorring(geo: Column): Column = Column {
    ST_ExteriorRing(Seq(geo.expr))
  }

  def st_scale(geo: Column, factorX: Column, factorY: Column): Column = Column {
    ST_Scale(Seq(geo.expr, factorX.expr, factorY.expr))
  }

  def st_affine(geo: Column, a: Column, b: Column, d: Column, e: Column, offsetX: Column, offsetY: Column): Column = Column {
    ST_Affine(Seq(geo.expr, a.expr, b.expr, d.expr, e.expr, offsetX.expr, offsetY.expr))
  }

  // Aggregate Function UDF API
  def st_union_aggr(geo: Column): Column = {
    val st_union_aggr_obj = new ST_Union_Aggr
    st_union_aggr_obj(geo)
  }

  // copy from spark
  private def withAggregateFunction(func: AggregateFunction,
                                    isDistinct: Boolean = false): Column = {
    Column(func.toAggregateExpression(isDistinct))
  }

  def st_envelope_aggr(geo: Column): Column = withAggregateFunction {
    EnvelopeAggr(geo.expr)
  }
}
