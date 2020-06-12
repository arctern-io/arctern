package org.apache.spark.sql.arctern

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.arctern.expressions._

object UdfRegistrator {
  def register(spark: SparkSession) = {
    // Register constructor UDFs
    spark.sessionState.functionRegistry.createOrReplaceTempFunction("ST_GeomFromText", ST_GeomFromText)
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
  }
}
