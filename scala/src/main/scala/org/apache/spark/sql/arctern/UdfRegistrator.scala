package org.apache.spark.sql.arctern

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.arctern.expressions._

object UdfRegistrator {
  def register(spark: SparkSession) = {
    spark.sessionState.functionRegistry.createOrReplaceTempFunction("ST_GeomFromText", ST_GeomFromText)
    spark.sessionState.functionRegistry.createOrReplaceTempFunction("ST_Within", ST_Within)
    spark.sessionState.functionRegistry.createOrReplaceTempFunction("ST_Centroid", ST_Centroid)
  }
}
