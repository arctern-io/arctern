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
  test("NearRoad") {
    val points = Seq(
      Row(new WKTReader().read("POINT(-100 -40)")),
      Row(new WKTReader().read("POINT(100 -40)")),
      Row(new WKTReader().read("POINT(-100 40)")),
      Row(new WKTReader().read("POINT(100 40)")),
      Row(new WKTReader().read("POINT(100 0)")),
    )

    val roads = Seq(
      Row(new WKTReader().read("POLYGON((-180 -90, 0 -90, 0 0, -180 0, -180 -90))")),
      Row(new WKTReader().read("POLYGON((0 -90, 180 -90, 180 0, 0 0, 0 -90))")),
      Row(new WKTReader().read("POLYGON((-180 0, 0 0, 0 90, -180 90, -180 0))")),
      Row(new WKTReader().read("POLYGON((0 0, 180 0, 180 90, 0 90, 0 0))")),
    )

    val pointSchema = StructType(Array(StructField("points", new GeometryUDT, nullable = false)))
    val roadSchema = StructType(Array(StructField("roads", new GeometryUDT, nullable = false)))

    val pointsDF = spark.createDataFrame(spark.sparkContext.parallelize(points), pointSchema)
    val roadsDF = spark.createDataFrame(spark.sparkContext.parallelize(roads), roadSchema)

    val rst = near_road(pointsDF, roadsDF)
    rst.show(false)
  }

  test("NearestRoad") {
    val points = Seq(
      Row(new WKTReader().read("POINT(-100 -40)")),
      Row(new WKTReader().read("POINT(100 -40)")),
      Row(new WKTReader().read("POINT(-100 40)")),
      Row(new WKTReader().read("POINT(100 40)")),
      Row(new WKTReader().read("POINT(100 0)")),
    )

    val roads = Seq(
      Row(new WKTReader().read("POLYGON((-180 -90, 0 -90, 0 0, -180 0, -180 -90))")),
      Row(new WKTReader().read("POLYGON((0 -90, 180 -90, 180 0, 0 0, 0 -90))")),
      Row(new WKTReader().read("POLYGON((-180 0, 0 0, 0 90, -180 90, -180 0))")),
      Row(new WKTReader().read("POLYGON((0 0, 180 0, 180 90, 0 90, 0 0))")),
    )

    val pointSchema = StructType(Array(StructField("points", new GeometryUDT, nullable = false)))
    val roadSchema = StructType(Array(StructField("roads", new GeometryUDT, nullable = false)))

    val pointsDF = spark.createDataFrame(spark.sparkContext.parallelize(points), pointSchema)
    val roadsDF = spark.createDataFrame(spark.sparkContext.parallelize(roads), roadSchema)

    val rst = nearest_road(pointsDF, roadsDF)
    rst.show(false)
  }

  test("NearestLocationOnRoad") {
    val points = Seq(
      Row(new WKTReader().read("POINT(-100 -40)")),
      Row(new WKTReader().read("POINT(100 -40)")),
      Row(new WKTReader().read("POINT(-100 40)")),
      Row(new WKTReader().read("POINT(100 40)")),
      Row(new WKTReader().read("POINT(100 0)")),
    )

    val roads = Seq(
      Row(new WKTReader().read("POLYGON((-180 -90, 0 -90, 0 0, -180 0, -180 -90))")),
      Row(new WKTReader().read("POLYGON((0 -90, 180 -90, 180 0, 0 0, 0 -90))")),
      Row(new WKTReader().read("POLYGON((-180 0, 0 0, 0 90, -180 90, -180 0))")),
      Row(new WKTReader().read("POLYGON((0 0, 180 0, 180 90, 0 90, 0 0))")),
    )

    val pointSchema = StructType(Array(StructField("points", new GeometryUDT, nullable = false)))
    val roadSchema = StructType(Array(StructField("roads", new GeometryUDT, nullable = false)))

    val pointsDF = spark.createDataFrame(spark.sparkContext.parallelize(points), pointSchema)
    val roadsDF = spark.createDataFrame(spark.sparkContext.parallelize(roads), roadSchema)

    val rst = nearest_location_on_road(pointsDF, roadsDF)
    rst.show(false)
  }
}
