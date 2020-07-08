package org.apache.spark.sql.arctern

import org.apache.spark.sql.arctern.expressions.ST_Within
import org.apache.spark.sql.arctern.functions._
import org.apache.spark.sql.arctern.index.RTreeIndex
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{Column, DataFrame, SparkSession}
import org.apache.zookeeper.KeeperException.UnimplementedException
import org.locationtech.jts.geom.{Envelope, Geometry}


import scala.collection.mutable

case class SpatialJoin(spark: SparkSession,
                       left: DataFrame,
                       right: DataFrame,
                       leftGeom: Column,
                       rightGeom: Column,
                       leftPrefix: String = "left",
                       rightPrefix: String = "right") {
  def apply(): DataFrame = {
    val temp_prefix = "arctern__"
    import spark.implicits._
    //    val points = left.select(monotonically_increasing_id().as("attr"), leftGeom.as("points"))
    //    val polygons = right.select(monotonically_increasing_id().as("id"), rightGeom.as("polygons"))

    val leftIndex = col(temp_prefix + "attr")
    val rightIndex = col(temp_prefix + "id")
    val candidateIndex = col(temp_prefix + "candidate")

    val points = left.withColumn(leftIndex.toString(), monotonically_increasing_id())
    val polygons = right.withColumn(rightIndex.toString(), monotonically_increasing_id())

    val index_data = polygons.select(rightIndex,
      st_envelopinternal(rightGeom).as("envs"))

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

    val cross = points.select(leftIndex, leftGeom).as[(Long, Geometry)].flatMap {
      tp => {
        val (attr, point) = tp
        val env = point.getEnvelopeInternal
        val polyList = broadcast.value.query(env)
        polyList.toArray.map(poly => (poly.asInstanceOf[Long], attr, point))
      }
    }.withColumnRenamed("_1", rightIndex.toString())
      .withColumnRenamed("_2", leftIndex.toString())
      .withColumnRenamed("_3", leftGeom.toString())
    cross.show()
    val joinResult =  cross.join(polygons, rightIndex.toString()).filter(st_within(leftGeom, rightGeom))

    joinResult.show()
    print(joinResult)
    val polygons_rows =

//    val result = cross.join().flatMap {
//      tp => {
//        val polygonId = tp._1
//        val ((pointId, point), polygon) = tp._2
//        if (point.within(polygon)) {
//          (pointId, polygonId) :: Nil
//        } else {
//          Nil
//        }
//      }
//    }
//    result.toDF()
    throw new UnimplementedException
    left
  }
}
