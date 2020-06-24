package org.apache.spark.sql.arctern

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.arctern.expressions._

object UdfRegistrator {
  def register(spark: SparkSession) = {
    // Register constructor UDFs
    spark.sessionState.functionRegistry.createOrReplaceTempFunction("ST_GeomFromText", ST_GeomFromText)
    spark.sessionState.functionRegistry.createOrReplaceTempFunction("ST_GeomFromWKB", ST_GeomFromWKB)
    spark.sessionState.functionRegistry.createOrReplaceTempFunction("ST_Point", ST_Point)
    spark.sessionState.functionRegistry.createOrReplaceTempFunction("ST_PolygonFromEnvelope", ST_PolygonFromEnvelope)
    spark.sessionState.functionRegistry.createOrReplaceTempFunction("ST_GeomFromGeoJSON", ST_GeomFromGeoJSON)
    spark.sessionState.functionRegistry.createOrReplaceTempFunction("ST_AsText", ST_AsText)
    spark.sessionState.functionRegistry.createOrReplaceTempFunction("ST_AsGeoJSON", ST_AsGeoJSON)
    // Register function UDFs
    spark.sessionState.functionRegistry.createOrReplaceTempFunction("ST_Within", ST_Within)
    spark.sessionState.functionRegistry.createOrReplaceTempFunction("ST_Centroid", ST_Centroid)
    spark.sessionState.functionRegistry.createOrReplaceTempFunction("ST_IsValid", ST_IsValid)
    spark.sessionState.functionRegistry.createOrReplaceTempFunction("ST_GeometryType", ST_GeometryType)
    spark.sessionState.functionRegistry.createOrReplaceTempFunction("ST_IsSimple", ST_IsSimple)
    spark.sessionState.functionRegistry.createOrReplaceTempFunction("ST_NPoints", ST_NPoints)
    spark.sessionState.functionRegistry.createOrReplaceTempFunction("ST_Envelope", ST_Envelope)
    spark.sessionState.functionRegistry.createOrReplaceTempFunction("ST_Buffer", ST_Buffer)
    spark.sessionState.functionRegistry.createOrReplaceTempFunction("ST_PrecisionReduce", ST_PrecisionReduce)
    spark.sessionState.functionRegistry.createOrReplaceTempFunction("ST_Intersection", ST_Intersection)
    spark.sessionState.functionRegistry.createOrReplaceTempFunction("ST_SimplifyPreserveTopology", ST_SimplifyPreserveTopology)
    spark.sessionState.functionRegistry.createOrReplaceTempFunction("ST_ConvexHull", ST_ConvexHull)
    spark.sessionState.functionRegistry.createOrReplaceTempFunction("ST_Area", ST_Area)
    spark.sessionState.functionRegistry.createOrReplaceTempFunction("ST_Length", ST_Length)
    spark.sessionState.functionRegistry.createOrReplaceTempFunction("ST_HausdorffDistance", ST_HausdorffDistance)
    spark.sessionState.functionRegistry.createOrReplaceTempFunction("ST_Distance", ST_Distance)
    spark.sessionState.functionRegistry.createOrReplaceTempFunction("ST_Equals", ST_Equals)
    spark.sessionState.functionRegistry.createOrReplaceTempFunction("ST_Touches", ST_Touches)
    spark.sessionState.functionRegistry.createOrReplaceTempFunction("ST_Overlaps", ST_Overlaps)
    spark.sessionState.functionRegistry.createOrReplaceTempFunction("ST_Crosses", ST_Crosses)
    spark.sessionState.functionRegistry.createOrReplaceTempFunction("ST_Contains", ST_Contains)
    spark.sessionState.functionRegistry.createOrReplaceTempFunction("ST_Intersects", ST_Intersects)
    spark.sessionState.functionRegistry.createOrReplaceTempFunction("ST_DistanceSphere", ST_DistanceSphere)
    spark.sessionState.functionRegistry.createOrReplaceTempFunction("ST_Transform", ST_Transform)
    spark.sessionState.functionRegistry.createOrReplaceTempFunction("ST_MakeValid", ST_MakeValid)
    spark.sessionState.functionRegistry.createOrReplaceTempFunction("ST_CurveToLine", ST_CurveToLine)
    spark.sessionState.functionRegistry.createOrReplaceTempFunction("ST_Translate", ST_Translate)
    spark.sessionState.functionRegistry.createOrReplaceTempFunction("ST_Rotate", ST_Rotate)
    spark.sessionState.functionRegistry.createOrReplaceTempFunction("ST_SymDifference", ST_SymDifference)
    spark.sessionState.functionRegistry.createOrReplaceTempFunction("ST_Difference", ST_Difference)
    spark.sessionState.functionRegistry.createOrReplaceTempFunction("ST_Union", ST_Union)
    spark.sessionState.functionRegistry.createOrReplaceTempFunction("ST_Disjoint", ST_Disjoint)
    spark.sessionState.functionRegistry.createOrReplaceTempFunction("ST_IsEmpty", ST_IsEmpty)
    spark.sessionState.functionRegistry.createOrReplaceTempFunction("ST_Boundary", ST_Boundary)
    spark.sessionState.functionRegistry.createOrReplaceTempFunction("ST_ExteriorRing", ST_ExteriorRing)
    spark.sessionState.functionRegistry.createOrReplaceTempFunction("ST_Scale", ST_Scale)
    spark.sessionState.functionRegistry.createOrReplaceTempFunction("ST_Affine", ST_Affine)
    // Register aggregate function UDFs
    spark.udf.register("ST_Union_Aggr", new ST_Union_Aggr)
    spark.sessionState.functionRegistry.createOrReplaceTempFunction("ST_Envelope_Aggr", seqs => EnvelopeAggr(seqs(0)))
  }
}
