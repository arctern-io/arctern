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
import org.apache.spark.sql.Row
import org.apache.spark.sql.arctern.GeometryUDT
import org.apache.spark.sql.types.{StructField, StructType}
import org.apache.spark.sql.arctern.functions._
import org.locationtech.jts.io.WKTReader

class MapMatchingTest extends AdapterTest {
  val points = Seq(
    Row(new WKTReader().read("POINT (-73.961003 40.760594)")),
    Row(new WKTReader().read("POINT (-73.959908 40.776353)")),
    Row(new WKTReader().read("POINT (-73.955183 40.773459)")),
    Row(new WKTReader().read("POINT (-73.985233 40.744682)")),
    Row(new WKTReader().read("POINT (-73.997969 40.682816)")),
    Row(new WKTReader().read("POINT (-73.996458 40.758197)")),
    Row(new WKTReader().read("POINT (-73.988240 40.748960)")),
    Row(new WKTReader().read("POINT (-73.985185 40.735828)")),
    Row(new WKTReader().read("POINT (-73.989726 40.767795)")),
    Row(new WKTReader().read("POINT (-73.992669 40.768327)")),
  )

  val roads = Seq(
    Row(new WKTReader().read("LINESTRING (-73.9975944 40.7140611, -73.9974922 40.7139962)")),
    Row(new WKTReader().read("LINESTRING (-73.9980065 40.7138119, -73.9980743 40.7137811)")),
    Row(new WKTReader().read("LINESTRING (-73.9975554 40.7141073, -73.9975944 40.7140611)")),
    Row(new WKTReader().read("LINESTRING (-73.9978864 40.7143170, -73.9976740 40.7140968)")),
    Row(new WKTReader().read("LINESTRING (-73.9979810 40.7136728, -73.9980743 40.7137811)")),
    Row(new WKTReader().read("LINESTRING (-73.9980743 40.7137811, -73.9984728 40.7136003)")),
    Row(new WKTReader().read("LINESTRING (-73.9611014 40.7608112, -73.9610636 40.7608639)")),
    Row(new WKTReader().read("LINESTRING (-73.9594166 40.7593773, -73.9593736 40.7593593)")),
    Row(new WKTReader().read("LINESTRING (-73.9616090 40.7602969, -73.9615014 40.7602517)")),
    Row(new WKTReader().read("LINESTRING (-73.9615569 40.7601753, -73.9615014 40.7602517)")),
  )

  test("NearRoad") {
    val pointSchema = StructType(Array(StructField("points", new GeometryUDT, nullable = false)))
    val roadSchema = StructType(Array(StructField("roads", new GeometryUDT, nullable = false)))

    val pointsDF = spark.createDataFrame(spark.sparkContext.parallelize(points), pointSchema)
    val roadsDF = spark.createDataFrame(spark.sparkContext.parallelize(roads), roadSchema)

    val rst = near_road(pointsDF, roadsDF, 1000)
    rst.show(false)
  }

  test("NearestRoad") {
    val pointSchema = StructType(Array(StructField("points", new GeometryUDT, nullable = false)))
    val roadSchema = StructType(Array(StructField("roads", new GeometryUDT, nullable = false)))

    val pointsDF = spark.createDataFrame(spark.sparkContext.parallelize(points), pointSchema)
    val roadsDF = spark.createDataFrame(spark.sparkContext.parallelize(roads), roadSchema)

    val rst = nearest_road(pointsDF, roadsDF)
    rst.show(false)
  }

  test("NearestLocationOnRoad") {
    val pointSchema = StructType(Array(StructField("points", new GeometryUDT, nullable = false)))
    val roadSchema = StructType(Array(StructField("roads", new GeometryUDT, nullable = false)))

    val pointsDF = spark.createDataFrame(spark.sparkContext.parallelize(points), pointSchema)
    val roadsDF = spark.createDataFrame(spark.sparkContext.parallelize(roads), roadSchema)

    val rst = nearest_location_on_road(pointsDF, roadsDF)
    rst.show(false)
  }
}
