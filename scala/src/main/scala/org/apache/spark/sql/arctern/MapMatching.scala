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
import org.apache.spark.sql.types.{StructField, StructType}
import org.locationtech.jts.geom.Geometry

object MapMatching {
  private def nearestRoad(geo: Geometry, index: Broadcast[RTreeIndex]): Geometry = {
    val env = geo.getEnvelopeInternal
    val results = index.value.query(env)
    var minDistance = 999999999.0
    var res: Geometry = null
    for (i <- 0 until results.size()) {
      val road = results.get(i).asInstanceOf[Geometry]
      val distance = geo.distance(road)
      if (minDistance > distance) {
        minDistance = distance
        res = road
      }
    }
    res
  }

  def mapMatching(points: DataFrame, roads: DataFrame): DataFrame = {
    val mm = new MapMatching
    mm.setPoints(points)
    mm.setRoads(roads)
    mm.buildIndex()
    mm.compute()
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

  def compute(): DataFrame = {
    val pointsRdd = points.rdd
    val broadcast = spark.sparkContext.broadcast(index)
    val rstRDD = pointsRdd.map(point => Row(MapMatching.nearestRoad(point.getAs[Geometry](0), broadcast)))
    val rstSchema = StructType(Array(StructField("mapMatching", new GeometryUDT, nullable = false)))
    spark.createDataFrame(rstRDD, rstSchema)
  }
}
