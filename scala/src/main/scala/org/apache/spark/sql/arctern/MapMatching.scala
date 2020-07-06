/*
 * Copyright (C) 2019-2020 Zilliz. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.spark.sql.arctern

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.sql.arctern.index.RTreeIndex
import org.apache.spark.sql.types.{BooleanType, StructField, StructType}
import org.locationtech.jts.geom.{Coordinate, Geometry}

object MapMatching {
  private def projection(x: Double, y: Double, x1: Double, y1: Double, x2: Double, y2: Double): Double = {
    val L2 = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1)
    if (L2 == 0.0) return Double.MaxValue
    val x1_x = x - x1
    val y1_y = y - y1
    val x1_x2 = x2 - x1
    val y1_y2 = y2 - y1
    var ratio = (x1_x * x1_x2 + y1_y * y1_y2) / L2
    ratio = if (ratio > 1) 1 else if (ratio < 0) 0 else ratio
    val prj_x = x1 + ratio * x1_x2
    val prj_y = y1 + ratio * y1_y2
    scala.math.sqrt((x - prj_x) * (x - prj_x) + (y - prj_y) * (y - prj_y))
  }

  private def compute(point: Geometry, road: Geometry): Double = {
    if (point.getGeometryType != "Point" || road.getGeometryType != "LineString") return Double.MaxValue
    val coordinates = road.getCoordinates
    val coordinate = point.getCoordinate
    var distance = Double.MaxValue
    for (i <- 0 until coordinates.size - 1) {
      val tmp = projection(coordinate.x, coordinate.y, coordinates(i).x, coordinates(i).y, coordinates(i + 1).x, coordinates(i + 1).y)
      if (tmp <= distance) distance = tmp
    }
    distance
  }

  private def computeNearestRoad(point: Geometry, index: Broadcast[RTreeIndex]): Geometry = {
    val env = point.getEnvelopeInternal
    val results = index.value.query(env)
    if (results.size() <= 0) return null
    var minDistance = Double.MaxValue
    var roadId: Int = -1
    for (i <- 0 until results.size()) {
      val road = results.get(i).asInstanceOf[Geometry]
      val distance = compute(point, road)
      if (distance <= minDistance) {
        minDistance = distance
        roadId = i
      }
    }
    results.get(roadId).asInstanceOf[Geometry]
  }

  private def computeNearRoad(geo: Geometry, index: Broadcast[RTreeIndex]): Boolean = {
    val env = geo.getEnvelopeInternal
    val results = index.value.query(env)
    results.size() > 0
  }
}

class MapMatching {
  private var roads: DataFrame = _

  private var points: DataFrame = _

  private val index: RTreeIndex = new RTreeIndex

  private val spark = SparkSession.builder().getOrCreate()

  private def setRoads(roads: DataFrame): Unit = this.roads = roads

  private def setPoints(points: DataFrame): Unit = this.points = points

  private def buildIndex(): Unit = {
    val roadArray = roads.coalesce(1).collect()
    for (road <- roadArray) {
      val roadGeometry = road.getAs[Geometry](0)
      index.insert(roadGeometry.getEnvelopeInternal, roadGeometry)
    }
  }

  def nearRoad(points: DataFrame, roads: DataFrame): DataFrame = {
    setPoints(points)
    setRoads(roads)
    buildIndex()
    val pointsRdd = points.rdd
    val broadcast = spark.sparkContext.broadcast(index)
    val rstRDD = pointsRdd.map(point => Row(MapMatching.computeNearRoad(point.getAs[Geometry](0), broadcast)))
    val rstSchema = StructType(Array(StructField("near_road", BooleanType, nullable = false)))
    spark.createDataFrame(rstRDD, rstSchema)
  }

  def nearestRoad(points: DataFrame, roads: DataFrame): DataFrame = {
    setPoints(points)
    setRoads(roads)
    buildIndex()
    val pointsRdd = points.rdd
    val broadcast = spark.sparkContext.broadcast(index)
    val rstRDD = pointsRdd.map(point => Row(MapMatching.computeNearestRoad(point.getAs[Geometry](0), broadcast)))
    val rstSchema = StructType(Array(StructField("near_road", new GeometryUDT, nullable = false)))
    spark.createDataFrame(rstRDD, rstSchema)
  }
}
