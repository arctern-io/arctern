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
import org.apache.spark.sql.arctern.functions._
import org.apache.spark.sql.types.{StructField, StructType}
import org.locationtech.jts.geom.Geometry
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

    val collect = rst.collect()
    assert(collect(0).getAs[Boolean](0) == true)
    assert(collect(1).getAs[Boolean](0) == false)
    assert(collect(2).getAs[Boolean](0) == false)
    assert(collect(3).getAs[Boolean](0) == false)
    assert(collect(4).getAs[Boolean](0) == false)
    assert(collect(5).getAs[Boolean](0) == false)
    assert(collect(6).getAs[Boolean](0) == false)
    assert(collect(7).getAs[Boolean](0) == false)
    assert(collect(8).getAs[Boolean](0) == false)
    assert(collect(9).getAs[Boolean](0) == false)
  }

  test("NearRoad-Null") {
    val pointSchema = StructType(Array(StructField("points", new GeometryUDT, nullable = true)))
    val roadSchema = StructType(Array(StructField("roads", new GeometryUDT, nullable = true)))

    val points = Seq(
      Row(null),
      Row(null),
      Row(null),
    )

    val roads = Seq(
      Row(null),
      Row(null),
    )

    val pointsDF = spark.createDataFrame(spark.sparkContext.parallelize(points), pointSchema)
    val roadsDF = spark.createDataFrame(spark.sparkContext.parallelize(roads), roadSchema)

    val rst = near_road(pointsDF, roadsDF, 1000)
    rst.show(false)

    val collect = rst.collect()
    assert(collect(0).getAs[Boolean](0) == false)
    assert(collect(1).getAs[Boolean](0) == false)
    assert(collect(2).getAs[Boolean](0) == false)

    val points2 = Seq(
      Row(null),
      Row(null),
      Row(null),
    )

    val roads2 = Seq(
      Row(new WKTReader().read("LINESTRING (-73.9616090 40.7602969, -73.9615014 40.7602517)")),
      Row(new WKTReader().read("LINESTRING (-73.9615569 40.7601753, -73.9615014 40.7602517)")),
    )

    val pointsDF2 = spark.createDataFrame(spark.sparkContext.parallelize(points2), pointSchema)
    val roadsDF2 = spark.createDataFrame(spark.sparkContext.parallelize(roads2), roadSchema)

    val rst2 = near_road(pointsDF2, roadsDF2, 1000)
    rst2.show(false)

    val collect2 = rst2.collect()
    assert(collect2(0).getAs[Boolean](0) == false)
    assert(collect2(1).getAs[Boolean](0) == false)
    assert(collect2(2).getAs[Boolean](0) == false)

    val points3 = Seq(
      Row(new WKTReader().read("POINT (-73.997969 40.682816)")),
      Row(new WKTReader().read("POINT (-73.996458 40.758197)")),
      Row(new WKTReader().read("POINT (-73.988240 40.748960)")),
      Row(new WKTReader().read("POINT (-73.985185 40.735828)")),
    )

    val roads3 = Seq(
      Row(null),
      Row(null),
    )

    val pointsDF3 = spark.createDataFrame(spark.sparkContext.parallelize(points3), pointSchema)
    val roadsDF3 = spark.createDataFrame(spark.sparkContext.parallelize(roads3), roadSchema)

    val rst3 = near_road(pointsDF3, roadsDF3, 1000)
    rst3.show(false)

    val collect3 = rst3.collect()
    assert(collect3(0).getAs[Boolean](0) == false)
    assert(collect3(1).getAs[Boolean](0) == false)
    assert(collect3(2).getAs[Boolean](0) == false)
    assert(collect3(3).getAs[Boolean](0) == false)

    val points4 = Seq(
      Row(null),
      Row(new WKTReader().read("POINT (-73.997969 40.682816)")),
      Row(new WKTReader().read("POINT (-73.996458 40.758197)")),
      Row(null),
      Row(new WKTReader().read("POINT (-73.988240 40.748960)")),
      Row(null),
      Row(new WKTReader().read("POINT (-73.985185 40.735828)")),
    )

    val roads4 = Seq(
      Row(null),
      Row(new WKTReader().read("LINESTRING (-73.9616090 40.7602969, -73.9615014 40.7602517)")),
      Row(new WKTReader().read("LINESTRING (-73.9615569 40.7601753, -73.9615014 40.7602517)")),
      Row(null),
    )

    val pointsDF4 = spark.createDataFrame(spark.sparkContext.parallelize(points4), pointSchema)
    val roadsDF4 = spark.createDataFrame(spark.sparkContext.parallelize(roads4), roadSchema)

    val rst4 = near_road(pointsDF4, roadsDF4, 1000)
    rst4.show(false)

    val collect4 = rst4.collect()
    assert(collect4(0).getAs[Boolean](0) == false)
    assert(collect4(1).getAs[Boolean](0) == false)
    assert(collect4(2).getAs[Boolean](0) == false)
    assert(collect4(3).getAs[Boolean](0) == false)
    assert(collect4(4).getAs[Boolean](0) == false)
    assert(collect4(5).getAs[Boolean](0) == false)
    assert(collect4(6).getAs[Boolean](0) == false)
  }

  test("NearRoad-Illegal-CRS") {
    val pointSchema = StructType(Array(StructField("points", new GeometryUDT, nullable = true)))
    val roadSchema = StructType(Array(StructField("roads", new GeometryUDT, nullable = true)))

    val points = Seq(
      Row(new WKTReader().read("POINT (-73.997969 40.682816)")),
      Row(new WKTReader().read("POINT (200 300)")),
      Row(new WKTReader().read("POINT (-73.996458 40.758197)")),
      Row(new WKTReader().read("POINT (876759574 83674745)")),
      Row(new WKTReader().read("POINT (-73.985185 40.735828)")),
      Row(new WKTReader().read("POINT (-0.00878457123 4868)")),
    )

    val roads = Seq(
      Row(new WKTReader().read("LINESTRING (-73.9979810 40.7136728, -73.9980743 40.7137811)")),
      Row(new WKTReader().read("LINESTRING (200 300, 500 900)")),
      Row(new WKTReader().read("LINESTRING (-73.9980743 40.7137811, -73.9984728 40.7136003)")),
      Row(new WKTReader().read("LINESTRING (-73.9611014 40.7608112, -73.9610636 40.7608639)")),
      Row(new WKTReader().read("LINESTRING (4547686 54657567, -85747389 -7846479)")),
      Row(new WKTReader().read("LINESTRING (-0.00004545 4868, 5677 4444)")),
      Row(new WKTReader().read("LINESTRING (-73.9616090 40.7602969, -73.9615014 40.7602517)")),
    )

    val pointsDF = spark.createDataFrame(spark.sparkContext.parallelize(points), pointSchema)
    val roadsDF = spark.createDataFrame(spark.sparkContext.parallelize(roads), roadSchema)

    val rst = near_road(pointsDF, roadsDF, 1000)
    rst.show(false)

    val collect = rst.collect()
    assert(collect(0).getAs[Boolean](0) == true)
    assert(collect(1).getAs[Boolean](0) == true)
    assert(collect(2).getAs[Boolean](0) == true)
    assert(collect(3).getAs[Boolean](0) == false)
    assert(collect(4).getAs[Boolean](0) == true)
    assert(collect(5).getAs[Boolean](0) == true)
  }

  test("NearRoad-Illegal-Geometry") {
    val pointSchema = StructType(Array(StructField("points", new GeometryUDT, nullable = true)))
    val roadSchema = StructType(Array(StructField("roads", new GeometryUDT, nullable = true)))

    val points = Seq(
      Row(new WKTReader().read("POINT (-73.997969 40.682816)")),
      Row(new WKTReader().read("LINESTRING (-73.996458 40.758197, -73.9980743 40.7137811)")),
      Row(new WKTReader().read("POLYGON ((-73.996458 40.758197, -73.896458 40.758197, -73.896458 40.858197, -73.996458 40.858197, -73.996458 40.758197))")),
    )

    val roads = Seq(
      Row(new WKTReader().read("POINT (-73.997969 40.682816)")),
      Row(new WKTReader().read("LINESTRING (-73.996458 40.758197, -73.9980743 40.7137811)")),
      Row(new WKTReader().read("POLYGON ((-73.996458 40.758197, -73.896458 40.758197, -73.896458 40.858197, -73.996458 40.858197, -73.996458 40.758197))")),
    )

    val pointsDF = spark.createDataFrame(spark.sparkContext.parallelize(points), pointSchema)
    val roadsDF = spark.createDataFrame(spark.sparkContext.parallelize(roads), roadSchema)

    val rst = near_road(pointsDF, roadsDF, 1000)
    rst.show(false)

    val collect = rst.collect()
    assert(collect(0).getAs[Boolean](0) == true)
    assert(collect(1).getAs[Boolean](0) == true)
    assert(collect(2).getAs[Boolean](0) == true)
  }

  test("NearestRoad") {
    val pointSchema = StructType(Array(StructField("points", new GeometryUDT, nullable = false)))
    val roadSchema = StructType(Array(StructField("roads", new GeometryUDT, nullable = false)))

    val pointsDF = spark.createDataFrame(spark.sparkContext.parallelize(points), pointSchema)
    val roadsDF = spark.createDataFrame(spark.sparkContext.parallelize(roads), roadSchema)

    val rst = nearest_road(pointsDF, roadsDF)
    rst.show(false)

    val collect = rst.collect()
    assert(collect(0).getAs[Geometry](0).toText == "LINESTRING (-73.9611014 40.7608112, -73.9610636 40.7608639)")
    assert(collect(1).getAs[Geometry](0).toText == "LINESTRING (-73.9611014 40.7608112, -73.9610636 40.7608639)")
    assert(collect(2).getAs[Geometry](0).toText == "LINESTRING (-73.9611014 40.7608112, -73.9610636 40.7608639)")
    assert(collect(3).getAs[Geometry](0).toText == "LINESTRING (-73.9615569 40.7601753, -73.9615014 40.7602517)")
    assert(collect(4).getAs[Geometry](0).toText == "LINESTRING (-73.9980743 40.7137811, -73.9984728 40.7136003)")
    assert(collect(5).getAs[Geometry](0).toText == "LINESTRING (-73.961609 40.7602969, -73.9615014 40.7602517)")
    assert(collect(6).getAs[Geometry](0).toText == "LINESTRING (-73.961609 40.7602969, -73.9615014 40.7602517)")
    assert(collect(7).getAs[Geometry](0).toText == "LINESTRING (-73.9978864 40.714317, -73.997674 40.7140968)")
    assert(collect(8).getAs[Geometry](0).toText == "LINESTRING (-73.961609 40.7602969, -73.9615014 40.7602517)")
    assert(collect(9).getAs[Geometry](0).toText == "LINESTRING (-73.961609 40.7602969, -73.9615014 40.7602517)")
  }

  test("NearestRoad-Null") {
    val pointSchema = StructType(Array(StructField("points", new GeometryUDT, nullable = true)))
    val roadSchema = StructType(Array(StructField("roads", new GeometryUDT, nullable = true)))

    val points = Seq(
      Row(null),
      Row(null),
      Row(null),
    )

    val roads = Seq(
      Row(null),
      Row(null),
    )

    val pointsDF = spark.createDataFrame(spark.sparkContext.parallelize(points), pointSchema)
    val roadsDF = spark.createDataFrame(spark.sparkContext.parallelize(roads), roadSchema)

    val rst = nearest_road(pointsDF, roadsDF)
    rst.show(false)

    val collect = rst.collect()
    assert(collect(0).getAs[Geometry](0).toText == "LINESTRING EMPTY")
    assert(collect(1).getAs[Geometry](0).toText == "LINESTRING EMPTY")
    assert(collect(2).getAs[Geometry](0).toText == "LINESTRING EMPTY")

    val points2 = Seq(
      Row(null),
      Row(null),
      Row(null),
    )

    val roads2 = Seq(
      Row(new WKTReader().read("LINESTRING (-73.9616090 40.7602969, -73.9615014 40.7602517)")),
      Row(new WKTReader().read("LINESTRING (-73.9615569 40.7601753, -73.9615014 40.7602517)")),
    )

    val pointsDF2 = spark.createDataFrame(spark.sparkContext.parallelize(points2), pointSchema)
    val roadsDF2 = spark.createDataFrame(spark.sparkContext.parallelize(roads2), roadSchema)

    val rst2 = nearest_road(pointsDF2, roadsDF2)
    rst2.show(false)

    val collect2 = rst2.collect()
    assert(collect2(0).getAs[Geometry](0).toText == "LINESTRING EMPTY")
    assert(collect2(1).getAs[Geometry](0).toText == "LINESTRING EMPTY")
    assert(collect2(2).getAs[Geometry](0).toText == "LINESTRING EMPTY")

    val points3 = Seq(
      Row(new WKTReader().read("POINT (-73.997969 40.682816)")),
      Row(new WKTReader().read("POINT (-73.996458 40.758197)")),
      Row(new WKTReader().read("POINT (-73.988240 40.748960)")),
      Row(new WKTReader().read("POINT (-73.985185 40.735828)")),
    )

    val roads3 = Seq(
      Row(null),
      Row(null),
    )

    val pointsDF3 = spark.createDataFrame(spark.sparkContext.parallelize(points3), pointSchema)
    val roadsDF3 = spark.createDataFrame(spark.sparkContext.parallelize(roads3), roadSchema)

    val rst3 = nearest_road(pointsDF3, roadsDF3)
    rst3.show(false)

    val collect3 = rst3.collect()
    assert(collect3(0).getAs[Geometry](0).toText == "LINESTRING EMPTY")
    assert(collect3(1).getAs[Geometry](0).toText == "LINESTRING EMPTY")
    assert(collect3(2).getAs[Geometry](0).toText == "LINESTRING EMPTY")
    assert(collect3(3).getAs[Geometry](0).toText == "LINESTRING EMPTY")

    val points4 = Seq(
      Row(null),
      Row(new WKTReader().read("POINT (-73.997969 40.682816)")),
      Row(new WKTReader().read("POINT (-73.996458 40.758197)")),
      Row(null),
      Row(new WKTReader().read("POINT (-73.988240 40.748960)")),
      Row(null),
      Row(new WKTReader().read("POINT (-73.985185 40.735828)")),
    )

    val roads4 = Seq(
      Row(null),
      Row(new WKTReader().read("LINESTRING (-73.9616090 40.7602969, -73.9615014 40.7602517)")),
      Row(new WKTReader().read("LINESTRING (-73.9615569 40.7601753, -73.9615014 40.7602517)")),
      Row(null),
    )

    val pointsDF4 = spark.createDataFrame(spark.sparkContext.parallelize(points4), pointSchema)
    val roadsDF4 = spark.createDataFrame(spark.sparkContext.parallelize(roads4), roadSchema)

    val rst4 = nearest_road(pointsDF4, roadsDF4)
    rst4.show(false)

    val collect4 = rst4.collect()
    assert(collect4(0).getAs[Geometry](0).toText == "LINESTRING EMPTY")
    assert(collect4(1).getAs[Geometry](0).toText == "LINESTRING (-73.9615569 40.7601753, -73.9615014 40.7602517)")
    assert(collect4(2).getAs[Geometry](0).toText == "LINESTRING (-73.961609 40.7602969, -73.9615014 40.7602517)")
    assert(collect4(3).getAs[Geometry](0).toText == "LINESTRING EMPTY")
    assert(collect4(4).getAs[Geometry](0).toText == "LINESTRING (-73.961609 40.7602969, -73.9615014 40.7602517)")
    assert(collect4(5).getAs[Geometry](0).toText == "LINESTRING EMPTY")
    assert(collect4(6).getAs[Geometry](0).toText == "LINESTRING (-73.9615569 40.7601753, -73.9615014 40.7602517)")
  }

  test("NearestRoad-Illegal-CRS") {
    val pointSchema = StructType(Array(StructField("points", new GeometryUDT, nullable = true)))
    val roadSchema = StructType(Array(StructField("roads", new GeometryUDT, nullable = true)))

    val points = Seq(
      Row(new WKTReader().read("POINT (-73.997969 40.682816)")),
      Row(new WKTReader().read("POINT (200 300)")),
      Row(new WKTReader().read("POINT (-73.996458 40.758197)")),
      Row(new WKTReader().read("POINT (876759574 83674745)")),
      Row(new WKTReader().read("POINT (-73.985185 40.735828)")),
      Row(new WKTReader().read("POINT (-0.00878457123 4868)")),
    )

    val roads = Seq(
      Row(new WKTReader().read("LINESTRING (-73.9979810 40.7136728, -73.9980743 40.7137811)")),
      Row(new WKTReader().read("LINESTRING (200 300, 500 900)")),
      Row(new WKTReader().read("LINESTRING (-73.9980743 40.7137811, -73.9984728 40.7136003)")),
      Row(new WKTReader().read("LINESTRING (-73.9611014 40.7608112, -73.9610636 40.7608639)")),
      Row(new WKTReader().read("LINESTRING (4547686 54657567, -85747389 -7846479)")),
      Row(new WKTReader().read("LINESTRING (-0.00004545 4868, 5677 4444)")),
      Row(new WKTReader().read("LINESTRING (-73.9616090 40.7602969, -73.9615014 40.7602517)")),
    )

    val pointsDF = spark.createDataFrame(spark.sparkContext.parallelize(points), pointSchema)
    val roadsDF = spark.createDataFrame(spark.sparkContext.parallelize(roads), roadSchema)

    val rst = nearest_road(pointsDF, roadsDF)
    rst.show(false)

    val collect = rst.collect()
    assert(collect(0).getAs[Geometry](0).toText == "LINESTRING (4547686 54657567, -85747389 -7846479)")
    assert(collect(1).getAs[Geometry](0).toText == "LINESTRING (200 300, 500 900)")
    assert(collect(2).getAs[Geometry](0).toText == "LINESTRING (4547686 54657567, -85747389 -7846479)")
    assert(collect(3).getAs[Geometry](0).toText == "LINESTRING EMPTY")
    assert(collect(4).getAs[Geometry](0).toText == "LINESTRING (4547686 54657567, -85747389 -7846479)")
    assert(collect(5).getAs[Geometry](0).toText == "LINESTRING (4547686 54657567, -85747389 -7846479)")
  }

  test("NearestRoad-Illegal-Geometry") {
    val pointSchema = StructType(Array(StructField("points", new GeometryUDT, nullable = true)))
    val roadSchema = StructType(Array(StructField("roads", new GeometryUDT, nullable = true)))

    val points = Seq(
      Row(new WKTReader().read("POINT (-73.997969 40.682816)")),
      Row(new WKTReader().read("LINESTRING (-73.996458 40.758197, -73.9980743 40.7137811)")),
      Row(new WKTReader().read("POLYGON ((-73.996458 40.758197, -73.896458 40.758197, -73.896458 40.858197, -73.996458 40.858197, -73.996458 40.758197))")),
    )

    val roads = Seq(
      Row(new WKTReader().read("POINT (-73.997969 40.682816)")),
      Row(new WKTReader().read("LINESTRING (-73.996458 40.758197, -73.9980743 40.7137811)")),
      Row(new WKTReader().read("POLYGON ((-73.996458 40.758197, -73.896458 40.758197, -73.896458 40.858197, -73.996458 40.858197, -73.996458 40.758197))")),
    )

    val pointsDF = spark.createDataFrame(spark.sparkContext.parallelize(points), pointSchema)
    val roadsDF = spark.createDataFrame(spark.sparkContext.parallelize(roads), roadSchema)

    val rst = nearest_road(pointsDF, roadsDF)
    rst.show(false)

    val collect = rst.collect()
    assert(collect(0).getAs[Geometry](0).toText == "LINESTRING EMPTY")
    assert(collect(1).getAs[Geometry](0).toText == "LINESTRING EMPTY")
    assert(collect(2).getAs[Geometry](0).toText == "LINESTRING EMPTY")
  }

  test("NearestLocationOnRoad") {
    val pointSchema = StructType(Array(StructField("points", new GeometryUDT, nullable = false)))
    val roadSchema = StructType(Array(StructField("roads", new GeometryUDT, nullable = false)))

    val pointsDF = spark.createDataFrame(spark.sparkContext.parallelize(points), pointSchema)
    val roadsDF = spark.createDataFrame(spark.sparkContext.parallelize(roads), roadSchema)

    val rst = nearest_location_on_road(pointsDF, roadsDF)
    rst.show(false)

    val collect = rst.collect()
    assert(collect(0).getAs[Geometry](0).toText == "POINT (-73.9611014 40.7608112)")
    assert(collect(1).getAs[Geometry](0).toText == "POINT (-73.9610636 40.7608639)")
    assert(collect(2).getAs[Geometry](0).toText == "POINT (-73.9610636 40.7608639)")
    assert(collect(3).getAs[Geometry](0).toText == "POINT (-73.9615569 40.7601753)")
    assert(collect(4).getAs[Geometry](0).toText == "POINT (-73.9984728 40.7136003)")
    assert(collect(5).getAs[Geometry](0).toText == "POINT (-73.961609 40.7602969)")
    assert(collect(6).getAs[Geometry](0).toText == "POINT (-73.961609 40.7602969)")
    assert(collect(7).getAs[Geometry](0).toText == "POINT (-73.9978864 40.714317)")
    assert(collect(8).getAs[Geometry](0).toText == "POINT (-73.961609 40.7602969)")
    assert(collect(9).getAs[Geometry](0).toText == "POINT (-73.961609 40.7602969)")
  }

  test("NearestLocationOnRoad-Null") {
    val pointSchema = StructType(Array(StructField("points", new GeometryUDT, nullable = true)))
    val roadSchema = StructType(Array(StructField("roads", new GeometryUDT, nullable = true)))

    val points = Seq(
      Row(null),
      Row(null),
      Row(null),
    )

    val roads = Seq(
      Row(null),
      Row(null),
    )

    val pointsDF = spark.createDataFrame(spark.sparkContext.parallelize(points), pointSchema)
    val roadsDF = spark.createDataFrame(spark.sparkContext.parallelize(roads), roadSchema)

    val rst = nearest_location_on_road(pointsDF, roadsDF)
    rst.show(false)

    val collect = rst.collect()
    assert(collect(0).getAs[Geometry](0).toText == "GEOMETRYCOLLECTION EMPTY")
    assert(collect(1).getAs[Geometry](0).toText == "GEOMETRYCOLLECTION EMPTY")
    assert(collect(2).getAs[Geometry](0).toText == "GEOMETRYCOLLECTION EMPTY")

    val points2 = Seq(
      Row(null),
      Row(null),
      Row(null),
    )

    val roads2 = Seq(
      Row(new WKTReader().read("LINESTRING (-73.9616090 40.7602969, -73.9615014 40.7602517)")),
      Row(new WKTReader().read("LINESTRING (-73.9615569 40.7601753, -73.9615014 40.7602517)")),
    )

    val pointsDF2 = spark.createDataFrame(spark.sparkContext.parallelize(points2), pointSchema)
    val roadsDF2 = spark.createDataFrame(spark.sparkContext.parallelize(roads2), roadSchema)

    val rst2 = nearest_location_on_road(pointsDF2, roadsDF2)
    rst2.show(false)

    val collect2 = rst2.collect()
    assert(collect2(0).getAs[Geometry](0).toText == "GEOMETRYCOLLECTION EMPTY")
    assert(collect2(1).getAs[Geometry](0).toText == "GEOMETRYCOLLECTION EMPTY")
    assert(collect2(2).getAs[Geometry](0).toText == "GEOMETRYCOLLECTION EMPTY")

    val points3 = Seq(
      Row(new WKTReader().read("POINT (-73.997969 40.682816)")),
      Row(new WKTReader().read("POINT (-73.996458 40.758197)")),
      Row(new WKTReader().read("POINT (-73.988240 40.748960)")),
      Row(new WKTReader().read("POINT (-73.985185 40.735828)")),
    )

    val roads3 = Seq(
      Row(null),
      Row(null),
    )

    val pointsDF3 = spark.createDataFrame(spark.sparkContext.parallelize(points3), pointSchema)
    val roadsDF3 = spark.createDataFrame(spark.sparkContext.parallelize(roads3), roadSchema)

    val rst3 = nearest_location_on_road(pointsDF3, roadsDF3)
    rst3.show(false)

    val collect3 = rst3.collect()
    assert(collect3(0).getAs[Geometry](0).toText == "GEOMETRYCOLLECTION EMPTY")
    assert(collect3(1).getAs[Geometry](0).toText == "GEOMETRYCOLLECTION EMPTY")
    assert(collect3(2).getAs[Geometry](0).toText == "GEOMETRYCOLLECTION EMPTY")
    assert(collect3(3).getAs[Geometry](0).toText == "GEOMETRYCOLLECTION EMPTY")

    val points4 = Seq(
      Row(null),
      Row(new WKTReader().read("POINT (-73.997969 40.682816)")),
      Row(new WKTReader().read("POINT (-73.996458 40.758197)")),
      Row(null),
      Row(new WKTReader().read("POINT (-73.988240 40.748960)")),
      Row(null),
      Row(new WKTReader().read("POINT (-73.985185 40.735828)")),
    )

    val roads4 = Seq(
      Row(null),
      Row(new WKTReader().read("LINESTRING (-73.9616090 40.7602969, -73.9615014 40.7602517)")),
      Row(new WKTReader().read("LINESTRING (-73.9615569 40.7601753, -73.9615014 40.7602517)")),
      Row(null),
    )

    val pointsDF4 = spark.createDataFrame(spark.sparkContext.parallelize(points4), pointSchema)
    val roadsDF4 = spark.createDataFrame(spark.sparkContext.parallelize(roads4), roadSchema)

    val rst4 = nearest_location_on_road(pointsDF4, roadsDF4)
    rst4.show(false)

    val collect4 = rst4.collect()
    assert(collect4(0).getAs[Geometry](0).toText == "GEOMETRYCOLLECTION EMPTY")
    assert(collect4(1).getAs[Geometry](0).toText == "POINT (-73.9615569 40.7601753)")
    assert(collect4(2).getAs[Geometry](0).toText == "POINT (-73.961609 40.7602969)")
    assert(collect4(3).getAs[Geometry](0).toText == "GEOMETRYCOLLECTION EMPTY")
    assert(collect4(4).getAs[Geometry](0).toText == "POINT (-73.961609 40.7602969)")
    assert(collect4(5).getAs[Geometry](0).toText == "GEOMETRYCOLLECTION EMPTY")
    assert(collect4(6).getAs[Geometry](0).toText == "POINT (-73.9615569 40.7601753)")
  }

  test("NearestLocationOnRoad-Illegal-CRS") {
    val pointSchema = StructType(Array(StructField("points", new GeometryUDT, nullable = true)))
    val roadSchema = StructType(Array(StructField("roads", new GeometryUDT, nullable = true)))

    val points = Seq(
      Row(new WKTReader().read("POINT (-73.997969 40.682816)")),
      Row(new WKTReader().read("POINT (200 300)")),
      Row(new WKTReader().read("POINT (-73.996458 40.758197)")),
      Row(new WKTReader().read("POINT (876759574 83674745)")),
      Row(new WKTReader().read("POINT (-73.985185 40.735828)")),
      Row(new WKTReader().read("POINT (-0.00878457123 4868)")),
    )

    val roads = Seq(
      Row(new WKTReader().read("LINESTRING (-73.9979810 40.7136728, -73.9980743 40.7137811)")),
      Row(new WKTReader().read("LINESTRING (200 300, 500 900)")),
      Row(new WKTReader().read("LINESTRING (-73.9980743 40.7137811, -73.9984728 40.7136003)")),
      Row(new WKTReader().read("LINESTRING (-73.9611014 40.7608112, -73.9610636 40.7608639)")),
      Row(new WKTReader().read("LINESTRING (4547686 54657567, -85747389 -7846479)")),
      Row(new WKTReader().read("LINESTRING (-0.00004545 4868, 5677 4444)")),
      Row(new WKTReader().read("LINESTRING (-73.9616090 40.7602969, -73.9615014 40.7602517)")),
    )

    val pointsDF = spark.createDataFrame(spark.sparkContext.parallelize(points), pointSchema)
    val roadsDF = spark.createDataFrame(spark.sparkContext.parallelize(roads), roadSchema)

    val rst = nearest_location_on_road(pointsDF, roadsDF)
    rst.show(false)

    val collect = rst.collect()
    assert(collect(0).getAs[Geometry](0).toText == "POINT (-24105432.46213577 34823308.77051425)")
    assert(collect(1).getAs[Geometry](0).toText == "POINT (200 300)")
    assert(collect(2).getAs[Geometry](0).toText == "POINT (-24105432.425837517 34823308.79564062)")
    assert(collect(3).getAs[Geometry](0).toText == "GEOMETRYCOLLECTION EMPTY")
    assert(collect(4).getAs[Geometry](0).toText == "POINT (-24105432.42868457 34823308.793669835)")
    assert(collect(5).getAs[Geometry](0).toText == "POINT (-24103123.3577811 34824907.17845403)")
  }

  test("NearestLocationOnRoad-Illegal-Geometry") {
    val pointSchema = StructType(Array(StructField("points", new GeometryUDT, nullable = true)))
    val roadSchema = StructType(Array(StructField("roads", new GeometryUDT, nullable = true)))

    val points = Seq(
      Row(new WKTReader().read("POINT (-73.997969 40.682816)")),
      Row(new WKTReader().read("LINESTRING (-73.996458 40.758197, -73.9980743 40.7137811)")),
      Row(new WKTReader().read("POLYGON ((-73.996458 40.758197, -73.896458 40.758197, -73.896458 40.858197, -73.996458 40.858197, -73.996458 40.758197))")),
    )

    val roads = Seq(
      Row(new WKTReader().read("POINT (-73.997969 40.682816)")),
      Row(new WKTReader().read("LINESTRING (-73.996458 40.758197, -73.9980743 40.7137811)")),
      Row(new WKTReader().read("POLYGON ((-73.996458 40.758197, -73.896458 40.758197, -73.896458 40.858197, -73.996458 40.858197, -73.996458 40.758197))")),
    )

    val pointsDF = spark.createDataFrame(spark.sparkContext.parallelize(points), pointSchema)
    val roadsDF = spark.createDataFrame(spark.sparkContext.parallelize(roads), roadSchema)

    val rst = nearest_location_on_road(pointsDF, roadsDF)
    rst.show(false)

    val collect = rst.collect()
    assert(collect(0).getAs[Geometry](0).toText == "POINT (-999999999 -999999999)")
    assert(collect(1).getAs[Geometry](0).toText == "POINT (-999999999 -999999999)")
    assert(collect(2).getAs[Geometry](0).toText == "POINT (-999999999 -999999999)")
  }
}
