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

import java.util

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.sql.arctern.index.RTreeIndex
import org.apache.spark.sql.types.{BooleanType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.locationtech.jts.geom.{Coordinate, Envelope, Geometry, GeometryFactory}

object MapMatching {
  val defaultExpandValue = 100 // default expand value: 100 meters

  private def RAD2DEG(x: Double): Double = x * 180.0 / scala.math.Pi

  private def expandEnvelope(env: Envelope, expandValue: Double): Envelope = {
    val deg_distance = RAD2DEG(expandValue / 6371251.46)
    new Envelope(env.getMinX - deg_distance,
      env.getMaxX + deg_distance,
      env.getMinY - deg_distance,
      env.getMaxY + deg_distance)
  }

  private def envelopeCheck(env: Envelope): Boolean = env.getMinX >= -180 && env.getMaxX <= 180 && env.getMinY >= -90 && env.getMaxY <= 90

  private def mapMatchingQuery(point: Geometry, index: RTreeIndex): util.List[_] = {
    var ev = defaultExpandValue
    do {
      val env = expandEnvelope(point.getEnvelopeInternal, ev)
      if (!envelopeCheck(env)) return index.query(env)
      val rst = index.query(env)
      if (rst.size() > 0) return rst else ev *= 2
    } while (true)
    throw new Exception("Illegal operation in map matching query.")
  }

  private def projection(x: Double, y: Double, x1: Double, y1: Double, x2: Double, y2: Double): (Double, Double, Double) = {
    val L2 = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1)
    if (L2 == 0.0) return (Double.MaxValue, -999999999, -999999999)
    val x1_x = x - x1
    val y1_y = y - y1
    val x1_x2 = x2 - x1
    val y1_y2 = y2 - y1
    var ratio = (x1_x * x1_x2 + y1_y * y1_y2) / L2
    ratio = if (ratio > 1) 1 else if (ratio < 0) 0 else ratio
    val prj_x = x1 + ratio * x1_x2
    val prj_y = y1 + ratio * y1_y2
    (scala.math.sqrt((x - prj_x) * (x - prj_x) + (y - prj_y) * (y - prj_y)), prj_x, prj_y)
  }

  private def compute(point: Geometry, road: Geometry): (Double, Double, Double) = {
    if (point.getGeometryType != "Point" || road.getGeometryType != "LineString") return (Double.MaxValue, -999999999, -999999999)
    val coordinates = road.getCoordinates
    val coordinate = point.getCoordinate
    var distance = Double.MaxValue
    var x: Double = -999999999
    var y: Double = -999999999
    for (i <- 0 until coordinates.size - 1) {
      val tmp = projection(coordinate.x, coordinate.y, coordinates(i).x, coordinates(i).y, coordinates(i + 1).x, coordinates(i + 1).y)
      if (tmp._1 <= distance) {
        distance = tmp._1
        x = tmp._2
        y = tmp._3
      }
    }
    (distance, x, y)
  }

  private def computeNearRoad(point: Geometry, index: Broadcast[RTreeIndex], expandValue: Double): Boolean = {
    if (point == null) return false
    val env = expandEnvelope(point.getEnvelopeInternal, expandValue)
    val results = index.value.query(env)
    results.size() > 0
  }

  private def computeNearestRoad(point: Geometry, index: Broadcast[RTreeIndex]): Geometry = {
    if (point == null) return new GeometryFactory().createLineString()
    val results = mapMatchingQuery(point, index.value)
    if (results.size() <= 0) return new GeometryFactory().createLineString()
    var minDistance = Double.MaxValue
    var roadId: Int = -1
    for (i <- 0 until results.size()) {
      val road = results.get(i).asInstanceOf[Geometry]
      val rstProjection = compute(point, road)
      if (rstProjection._1 <= minDistance) {
        minDistance = rstProjection._1
        roadId = i
      }
    }
    if (minDistance == Double.MaxValue) return new GeometryFactory().createLineString()
    results.get(roadId).asInstanceOf[Geometry]
  }

  private def computeNearestLocationOnRoad(point: Geometry, index: Broadcast[RTreeIndex]): Geometry = {
    // Empty Points cannot be represented in WKB.
    // So here we use Empty GeometryCollection.
    if (point == null) return new GeometryFactory().createGeometryCollection()
    val results = mapMatchingQuery(point, index.value)
    if (results.size() <= 0) return new GeometryFactory().createGeometryCollection()
    var minDistance = Double.MaxValue
    var x: Double = -999999999
    var y: Double = -999999999
    for (i <- 0 until results.size()) {
      val road = results.get(i).asInstanceOf[Geometry]
      val rstProjection = compute(point, road)
      if (rstProjection._1 <= minDistance) {
        minDistance = rstProjection._1
        x = rstProjection._2
        y = rstProjection._3
      }
    }
    new GeometryFactory().createPoint(new Coordinate(x, y))
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
    val roadArray = roads.coalesce(numPartitions = 1).collect()
    for (road <- roadArray) {
      val roadGeometry = road.getAs[Geometry](0)
      if (roadGeometry != null) index.insert(roadGeometry.getEnvelopeInternal, roadGeometry)
    }
  }

  def nearRoad(points: DataFrame, roads: DataFrame, expandValue: Double): DataFrame = {
    setPoints(points)
    setRoads(roads)
    buildIndex()
    val pointsRdd = points.rdd
    val broadcast = spark.sparkContext.broadcast(index)
    val rstRDD = pointsRdd.map(point => Row(MapMatching.computeNearRoad(point.getAs[Geometry](0), broadcast, expandValue)))
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
    val rstSchema = StructType(Array(StructField("nearest_road", new GeometryUDT, nullable = false)))
    spark.createDataFrame(rstRDD, rstSchema)
  }

  def nearestLocationOnRoad(points: DataFrame, roads: DataFrame): DataFrame = {
    setPoints(points)
    setRoads(roads)
    buildIndex()
    val pointsRdd = points.rdd
    val broadcast = spark.sparkContext.broadcast(index)
    val rstRDD = pointsRdd.map(point => Row(MapMatching.computeNearestLocationOnRoad(point.getAs[Geometry](0), broadcast)))
    val rstSchema = StructType(Array(StructField("nearest_location_on_road", new GeometryUDT, nullable = false)))
    spark.createDataFrame(rstRDD, rstSchema)
  }
}
