package org.apache.spark.sql.arctern

import org.apache.spark.sql.arctern.functions.st_envelopinternal
import org.apache.spark.sql.arctern.index.RTreeIndex
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.locationtech.jts.geom.{Envelope, Geometry}



import scala.collection.mutable

case class WithinJoin(spark: SparkSession,
                      left: DataFrame, leftIndex: Symbol, leftGeom: Symbol,
                      right: DataFrame, rightIndex: Symbol, rightGeom: Symbol) {
  def apply(): DataFrame = {
    import spark.implicits._
    val points = left.select(leftIndex.as("attr"), leftGeom.as("points"))
    val polygons = right.select(rightIndex.as("id"), rightGeom.as("polygons"))
    val index_data = polygons.select(rightIndex.as("id"),
      st_envelopinternal('polygons).as("envs"))
    val tree = new RTreeIndex
    index_data.collect().foreach {
      row => {
        val id = row.getAs[Long](0)
        val data = row.getAs[mutable.WrappedArray[Double]](1)
        val env = new Envelope(data(0), data(2), data(1), data(3))
        tree.insert(env, id)
      }
    }

    val broadcast = spark.sparkContext.broadcast(tree)

    val points_rdd = points.as[(Long, Geometry)].rdd
    val polygon_rdd = polygons.as[(Long, Geometry)].rdd

    val cross = points_rdd.flatMap {
      tp => {
        val (attr, point) = tp
        val env = point.getEnvelopeInternal
        val polyList = broadcast.value.query(env)
        polyList.toArray.map(poly => (poly.asInstanceOf[Long], (attr, point)))
      }
    }

    val result = cross.join(polygon_rdd).flatMap {
      tp => {
        val polygonId = tp._1
        val ((pointId, point), polygon) = tp._2
        if (point.within(polygon)) {
          (pointId, polygonId) :: Nil
        } else {
          Nil
        }
      }
    }
    result.toDF(leftIndex.name, rightIndex.name)
  }
}
