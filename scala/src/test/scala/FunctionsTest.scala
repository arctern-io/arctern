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
import org.apache.spark.sql.types._
import org.apache.spark.sql.arctern._
import org.apache.spark.sql.arctern.expressions.api._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.functions.col

class FunctionsTest extends AdapterTest {
  test("ST_Within") {
    val data = Seq(
      Row(1, GeometryUDT.FromWkt("POINT (20 20)"), GeometryUDT.FromWkt("POLYGON ((0 0, 40 0, 40 40, 0 40, 0 0))")),
      Row(2, GeometryUDT.FromWkt("POINT (50 50)"), GeometryUDT.FromWkt("POLYGON ((0 0, 40 0, 40 40, 0 40, 0 0))")),
      Row(3, GeometryUDT.FromWkt("POLYGON ((10 10, 20 10, 20 20, 10 20, 10 10))"), GeometryUDT.FromWkt("POLYGON ((0 0, 40 0, 40 40, 0 40, 0 0))")),
      Row(4, GeometryUDT.FromWkt("POLYGON ((10 10, 50 10, 50 50, 10 50, 10 10))"), GeometryUDT.FromWkt("POLYGON ((0 0, 40 0, 40 40, 0 40, 0 0))"))
    )

    val rdd_d = spark.sparkContext.parallelize(data)
    val schema = StructType(Array(StructField("idx", IntegerType, nullable = false), StructField("geo1", new GeometryUDT, nullable = false), StructField("geo2", new GeometryUDT, nullable = false)))
    val df = spark.createDataFrame(rdd_d, schema)
    df.createOrReplaceTempView("data")
    val rst = spark.sql("select idx, ST_Within(geo1, geo2) from data")

    //    rst.queryExecution.debug.codegen()
    val collect = rst.collect()

    assert(collect(0).getBoolean(1) == true)
    assert(collect(1).getBoolean(1) == false)
    assert(collect(2).getBoolean(1) == true)
    assert(collect(3).getBoolean(1) == false)

    rst.show(false)

    val rst2 = df.select(st_within(col("geo1"), col("geo2")))
    rst2.show(false)
    val collect2 = rst2.collect()
  }

  test("ST_Within-Null") {
    val data = Seq(
      Row(1, "error geo", "POLYGON ((0 0, 40 0, 40 40, 0 40, 0 0))"),
      Row(2, "POINT (50 50)", "error geo"),
      Row(3, "error geo", "error geo"),
      Row(4, "POLYGON ((10 10, 50 10, 50 50, 10 50, 10 10))", "POLYGON ((0 0, 40 0, 40 40, 0 40, 0 0))")
    )

    val rdd_d = spark.sparkContext.parallelize(data)
    val schema = StructType(Array(StructField("idx", IntegerType, nullable = false), StructField("geo1", StringType, nullable = false), StructField("geo2", StringType, nullable = false)))
    val df = spark.createDataFrame(rdd_d, schema)
    df.createOrReplaceTempView("table_ST_Within")
    val rst = spark.sql("select idx, ST_Within(ST_GeomFromText(geo1), ST_GeomFromText(geo2)) from table_ST_Within")

    //    rst.queryExecution.debug.codegen()

    val collect = rst.collect()

    assert(collect(0).isNullAt(1))
    assert(collect(1).isNullAt(1))
    assert(collect(2).isNullAt(1))
    assert(collect(3).getBoolean(1) == false)

    rst.show(false)

    val rst2 = df.select(st_within(st_geomfromtext(col("geo1")), st_geomfromtext(col("geo2"))))
    rst2.show(false)
    val collect2 = rst2.collect()
  }

  test("ST_Centroid") {
    val data = Seq(
      Row(1, GeometryUDT.FromWkt("Polygon((0 0, 0 1, 1 1, 1 0, 0 0))")),
      Row(2, GeometryUDT.FromWkt("LINESTRING (0 0, 10 10, 20 20)"))
    )

    val schema = StructType(Array(StructField("idx", IntegerType, false), StructField("geo", new GeometryUDT, false)))
    val df = spark.createDataFrame(spark.sparkContext.parallelize(data), schema)
    df.createOrReplaceTempView("data")

    val rst = spark.sql("select idx, ST_Centroid(geo) from data ")
    //    rst.queryExecution.debug.codegen()
    rst.show(false)
    val collect = rst.collect()

    val rst2 = df.select(st_centroid(col("geo")))
    rst2.show(false)
    val collect2 = rst2.collect()
  }

  test("ST_Centroid-Null") {
    val data = Seq(
      Row(1, "Polygon(0 0, 0 1, 1 1, 1 0, 0 0)"),
      Row(2, "LINESTRING (0 0, 10 10, 20 20)"),
      Row(3, "error geo")
    )

    val schema = StructType(Array(StructField("idx", IntegerType, false), StructField("geo", StringType, false)))
    val df = spark.createDataFrame(spark.sparkContext.parallelize(data), schema)
    df.createOrReplaceTempView("data")

    val rst = spark.sql("select idx, ST_Centroid(ST_GeomFromText(geo)) from data ")
    //    rst.queryExecution.debug.codegen()
    val collect = rst.collect()
    assert(collect(0).isNullAt(1))
    assert(collect(2).isNullAt(1))

    val rst2 = df.select(st_centroid(st_geomfromtext(col("geo"))))
    rst2.show(false)
    val collect2 = rst2.collect()
  }

  test("ST_Within-Nest") {
    val data = Seq(
      Row(1, "polygon((0 0, 0 1,1 1, 1 0, 0 0))", "polygon((0 0, 0 1,1 1, 1 0, 0 0))"),
      Row(2, "error geo", "polygon((0 0, 0 1,1 1, 1 0, 0 0))"),
      Row(3, "polygon((0 0, 0 1,1 1, 1 0, 0 0))", "error geo")
    )
    val schema = StructType(Array(StructField("idx", IntegerType, false), StructField("geo1", StringType, false), StructField("geo2", StringType, false)))
    val df = spark.createDataFrame(spark.sparkContext.parallelize(data), schema)
    df.createOrReplaceTempView("data")
    val rst = spark.sql("select idx, st_within(st_centroid(ST_GeomFromText(geo1)), st_centroid(ST_GeomFromText(geo2))) from data")
    // rst.queryExecution.debug.codegen()
    val collect = rst.collect()
    assert(collect(0).getBoolean(1))
    assert(collect(1).isNullAt(1))
    assert(collect(2).isNullAt(1))
    rst.show(false)

    val rst2 = df.select(st_within(st_centroid(st_geomfromtext(col("geo1"))), st_centroid(st_geomfromtext(col("geo2")))))
    rst2.show(false)
    val collect2 = rst2.collect()
  }

  test("ST_IsValid") {
    val data = Seq(
      Row(GeometryUDT.FromWkt("Polygon((0 0, 0 1, 1 1, 1 0, 0 0))")),
      Row(GeometryUDT.FromWkt("LINESTRING (0 0, 10 10, 20 20)")),
      Row(GeometryUDT.FromWkt("POINT (0 0)"))
    )

    val schema = StructType(Array(StructField("geo", new GeometryUDT, nullable = true)))
    val df = spark.createDataFrame(spark.sparkContext.parallelize(data), schema)
    df.createOrReplaceTempView("data")

    val rst = spark.sql("select ST_IsValid(geo) from data ")

    //    rst.queryExecution.debug.codegen()

    rst.show(false)
    val collect = rst.collect()

    val rst2 = df.select(st_isvalid(col("geo")))
    rst2.show(false)
    val collect2 = rst2.collect()
  }

  test("ST_IsValid-Null") {
    val data = Seq(
      Row("error geo"),
      Row("LINESTRING (0 0, 10 10, 20 20)"),
      Row(null)
    )

    val schema = StructType(Array(StructField("geo", StringType, nullable = true)))
    val df = spark.createDataFrame(spark.sparkContext.parallelize(data), schema)
    df.createOrReplaceTempView("data")

    val rst = spark.sql("select ST_IsValid(ST_GeomFromText(geo)) from data ")
    val collect = rst.collect()

    //    rst.queryExecution.debug.codegen()

    assert(collect(0).isNullAt(0))
    assert(collect(2).isNullAt(0))

    rst.show(false)

    val rst2 = df.select(st_isvalid(st_geomfromtext(col("geo"))))
    rst2.show(false)
    val collect2 = rst2.collect()
  }

  test("ST_GeometryType") {
    val data = Seq(
      Row(GeometryUDT.FromWkt("Polygon((0 0, 0 1, 1 1, 1 0, 0 0))")),
      Row(GeometryUDT.FromWkt("LINESTRING (0 0, 10 10, 20 20)")),
      Row(GeometryUDT.FromWkt("POINT (0 0)")),
      Row(GeometryUDT.FromWkt("POLYGON ((1 2,3 4,5 6,1 2))")),
      Row(GeometryUDT.FromWkt("MULTIPOLYGON ( ((0 0, 1 0, 1 1,0 0)) )"))
    )

    val schema = StructType(Array(StructField("geo", new GeometryUDT, nullable = true)))
    val df = spark.createDataFrame(spark.sparkContext.parallelize(data), schema)
    df.createOrReplaceTempView("data")

    val rst = spark.sql("select ST_GeometryType(geo) from data ")

    //    rst.queryExecution.debug.codegen()

    rst.show(false)
    val collect = rst.collect()

    val rst2 = df.select(st_geometrytype(col("geo")))
    rst2.show(false)
    val collect2 = rst2.collect()
  }

  test("ST_GeometryType-Null") {
    val data = Seq(
      Row("error geo"),
      Row("LINESTRING (0 0, 10 10, 20 20)"),
      Row(null),
      Row("POLYGON ((1 2,3 4,5 6,1 2))"),
      Row("MULTIPOLYGON ( ((0 0, 1 0, 1 1,0 0)) )"),
    )

    val schema = StructType(Array(StructField("geo", StringType, nullable = true)))
    val df = spark.createDataFrame(spark.sparkContext.parallelize(data), schema)
    df.createOrReplaceTempView("data")

    val rst = spark.sql("select ST_GeometryType(ST_GeomFromText(geo)) from data ")
    val collect = rst.collect()

    //    rst.queryExecution.debug.codegen()

    assert(collect(0).isNullAt(0))
    assert(collect(2).isNullAt(0))

    rst.show(false)

    val rst2 = df.select(st_geometrytype(st_geomfromtext(col("geo"))))
    rst2.show(false)
    val collect2 = rst2.collect()
  }

  test("ST_IsSimple") {
    val data = Seq(
      Row(GeometryUDT.FromWkt("Polygon((0 0, 0 1, 1 1, 1 0, 0 0))")),
      Row(GeometryUDT.FromWkt("LINESTRING (0 0, 10 10, 20 20)")),
      Row(GeometryUDT.FromWkt("POINT (0 0)")),
      Row(GeometryUDT.FromWkt("POLYGON ((1 2,3 4,5 6,1 2))")),
      Row(GeometryUDT.FromWkt("MULTIPOLYGON ( ((0 0, 1 0, 1 1,0 0)) )"))
    )

    val schema = StructType(Array(StructField("geo", new GeometryUDT, nullable = true)))
    val df = spark.createDataFrame(spark.sparkContext.parallelize(data), schema)
    df.createOrReplaceTempView("data")

    val rst = spark.sql("select ST_IsSimple(geo) from data ")

    //    rst.queryExecution.debug.codegen()

    rst.show(false)
    val collect = rst.collect()

    val rst2 = df.select(st_issimple(col("geo")))
    rst2.show(false)
    val collect2 = rst2.collect()
  }

  test("ST_IsSimple-Null") {
    val data = Seq(
      Row("error geo"),
      Row("LINESTRING (0 0, 10 10, 20 20)"),
      Row(null),
      Row("POLYGON ((1 2,3 4,5 6,1 2))"),
      Row("MULTIPOLYGON ( ((0 0, 1 0, 1 1,0 0)) )"),
    )

    val schema = StructType(Array(StructField("geo", StringType, nullable = true)))
    val df = spark.createDataFrame(spark.sparkContext.parallelize(data), schema)
    df.createOrReplaceTempView("data")

    val rst = spark.sql("select ST_IsSimple(ST_GeomFromText(geo)) from data ")
    val collect = rst.collect()

    //    rst.queryExecution.debug.codegen()

    assert(collect(0).isNullAt(0))
    assert(collect(2).isNullAt(0))

    rst.show(false)

    val rst2 = df.select(st_centroid(st_geomfromtext(col("geo"))))
    rst2.show(false)
    val collect2 = rst2.collect()
  }

  test("ST_NPoints") {
    val data = Seq(
      Row(GeometryUDT.FromWkt("Polygon((0 0, 0 1, 1 1, 1 0, 0 0))")),
      Row(GeometryUDT.FromWkt("LINESTRING (0 0, 10 10, 20 20)")),
      Row(GeometryUDT.FromWkt("POINT (0 0)")),
      Row(GeometryUDT.FromWkt("POLYGON EMPTY")),
      Row(GeometryUDT.FromWkt("MULTIPOLYGON ( ((0 0, 1 0, 1 1,0 0)) )"))
    )

    val schema = StructType(Array(StructField("geo", new GeometryUDT, nullable = true)))
    val df = spark.createDataFrame(spark.sparkContext.parallelize(data), schema)
    df.createOrReplaceTempView("data")

    val rst = spark.sql("select ST_NPoints(geo) from data ")

    //    rst.queryExecution.debug.codegen()

    rst.show(false)
    val collect = rst.collect()

    val rst2 = df.select(st_npoints(col("geo")))
    rst2.show(false)
    val collect2 = rst2.collect()
  }

  test("ST_NPoints-Null") {
    val data = Seq(
      Row("error geo"),
      Row("LINESTRING (0 0, 10 10, 20 20)"),
      Row(null),
      Row("POLYGON ((1 2,3 4,5 6,1 2))"),
      Row("MULTIPOLYGON ( ((0 0, 1 0, 1 1,0 0)) )"),
    )

    val schema = StructType(Array(StructField("geo", StringType, nullable = true)))
    val df = spark.createDataFrame(spark.sparkContext.parallelize(data), schema)
    df.createOrReplaceTempView("data")

    val rst = spark.sql("select ST_NPoints(ST_GeomFromText(geo)) from data ")
    val collect = rst.collect()

    //    rst.queryExecution.debug.codegen()

    assert(collect(0).isNullAt(0))
    assert(collect(2).isNullAt(0))

    rst.show(false)

    val rst2 = df.select(st_npoints(st_geomfromtext(col("geo"))))
    rst2.show(false)
    val collect2 = rst2.collect()
  }

  test("ST_Envelope") {
    val data = Seq(
      Row(GeometryUDT.FromWkt("Polygon((0 0, 0 1, 1 1, 1 0, 0 0))")),
      Row(GeometryUDT.FromWkt("LINESTRING (0 0, 10 10, 20 20)")),
      Row(GeometryUDT.FromWkt("POINT (0 0)")),
      Row(GeometryUDT.FromWkt("MULTIPOLYGON ( ((0 0, 1 0, 1 1,0 0)) )"))
    )

    val schema = StructType(Array(StructField("geo", new GeometryUDT, nullable = true)))
    val df = spark.createDataFrame(spark.sparkContext.parallelize(data), schema)
    df.createOrReplaceTempView("data")

    val rst = spark.sql("select ST_Envelope(geo) from data ")

    //    rst.queryExecution.debug.codegen()

    rst.show(false)
    val collect = rst.collect()

    val rst2 = df.select(st_envelope(col("geo")))
    rst2.show(false)
    val collect2 = rst2.collect()
  }

  test("ST_Envelope-Null") {
    val data = Seq(
      Row("error geo"),
      Row("LINESTRING (0 0, 10 10, 20 20)"),
      Row(null),
      Row("POLYGON ((1 2,3 4,5 6,1 2))"),
      Row("MULTIPOLYGON ( ((0 0, 1 0, 1 1,0 0)) )"),
    )

    val schema = StructType(Array(StructField("geo", StringType, nullable = true)))
    val df = spark.createDataFrame(spark.sparkContext.parallelize(data), schema)
    df.createOrReplaceTempView("data")

    val rst = spark.sql("select ST_Envelope(ST_GeomFromText(geo)) from data ")
    val collect = rst.collect()

    //    rst.queryExecution.debug.codegen()

    assert(collect(0).isNullAt(0))
    assert(collect(2).isNullAt(0))

    rst.show(false)

    val rst2 = df.select(st_envelope(st_geomfromtext(col("geo"))))
    rst2.show(false)
    val collect2 = rst2.collect()
  }

  test("ST_Buffer") {
    val data = Seq(
      Row(GeometryUDT.FromWkt("Polygon((0 0, 0 1, 1 1, 1 0, 0 0))")),
      Row(GeometryUDT.FromWkt("LINESTRING (0 0, 10 10, 20 20)")),
      Row(GeometryUDT.FromWkt("POINT (0 0)")),
      Row(GeometryUDT.FromWkt("POLYGON EMPTY")),
      Row(GeometryUDT.FromWkt("MULTIPOLYGON ( ((0 0, 1 0, 1 1,0 0)) )"))
    )

    val schema = StructType(Array(StructField("geo", new GeometryUDT, nullable = true)))
    val df = spark.createDataFrame(spark.sparkContext.parallelize(data), schema)
    df.createOrReplaceTempView("data")

    val rst = spark.sql("select ST_Buffer(geo, 1) from data ")

    //    rst.queryExecution.debug.codegen()

    rst.show(false)
    val collect = rst.collect()

    val rst2 = df.select(st_buffer(col("geo"), lit(1)))
    rst2.show(false)
    val collect2 = rst2.collect()
  }

  test("ST_Buffer-Null") {
    val data = Seq(
      Row("error geo"),
      Row("LINESTRING (0 0, 10 10, 20 20)"),
      Row(null),
      Row("POLYGON ((1 2,3 4,5 6,1 2))"),
      Row("MULTIPOLYGON ( ((0 0, 1 0, 1 1,0 0)) )"),
    )

    val schema = StructType(Array(StructField("geo", StringType, nullable = true)))
    val df = spark.createDataFrame(spark.sparkContext.parallelize(data), schema)
    df.createOrReplaceTempView("data")

    val rst = spark.sql("select ST_Buffer(ST_GeomFromText(geo), 1) from data ")
    val collect = rst.collect()

    //    rst.queryExecution.debug.codegen()

    assert(collect(0).isNullAt(0))
    assert(collect(2).isNullAt(0))

    rst.show(false)

    val rst2 = df.select(st_buffer(st_geomfromtext(col("geo")), lit(1)))
    rst2.show(false)
    val collect2 = rst2.collect()
  }

  test("ST_PrecisionReduce") {
    val data = Seq(
      Row(GeometryUDT.FromWkt("Polygon((0.0001 0.0001, 0.0001 1.32435, 1.341312 1.32435, 1.341312 0.0001, 0.0001 0.0001))")),
      Row(GeometryUDT.FromWkt("LINESTRING (0.12 0.12, 10.234 10.456, 20.1 20.5566)")),
      Row(GeometryUDT.FromWkt("POINT (0.12345 0.346577)")),
      Row(GeometryUDT.FromWkt("POLYGON EMPTY")),
      Row(GeometryUDT.FromWkt("MULTIPOLYGON ( ((0.1234 0.1234, 1.1234 0.1234, 1.1234 1.1234,0.1234 0.1234)) )"))
    )

    val schema = StructType(Array(StructField("geo", new GeometryUDT, nullable = true)))
    val df = spark.createDataFrame(spark.sparkContext.parallelize(data), schema)
    df.createOrReplaceTempView("data")

    val rst = spark.sql("select ST_PrecisionReduce(geo, 2) from data ")

    //    rst.queryExecution.debug.codegen()

    rst.show(false)
    val collect = rst.collect()

    val rst2 = df.select(st_precisionreduce(col("geo"), lit(2)))
    rst2.show(false)
    val collect2 = rst2.collect()
  }

  test("ST_PrecisionReduce-Null") {
    val data = Seq(
      Row("error geo"),
      Row("LINESTRING (0.12 0.12, 10.234 10.456, 20.1 20.5566)"),
      Row(null),
      Row("POLYGON ((1 2,3 4,5 6,1 2))"),
      Row("MULTIPOLYGON ( ((0.1234 0.1234, 1.1234 0.1234, 1.1234 1.1234,0.1234 0.1234)) )"),
    )

    val schema = StructType(Array(StructField("geo", StringType, nullable = true)))
    val df = spark.createDataFrame(spark.sparkContext.parallelize(data), schema)
    df.createOrReplaceTempView("data")

    val rst = spark.sql("select ST_PrecisionReduce(ST_GeomFromText(geo), 2) from data ")
    val collect = rst.collect()

    //    rst.queryExecution.debug.codegen()

    assert(collect(0).isNullAt(0))
    assert(collect(2).isNullAt(0))

    rst.show(false)

    val rst2 = df.select(st_precisionreduce(st_geomfromtext(col("geo")), lit(2)))
    rst2.show(false)
    val collect2 = rst2.collect()
  }

  test("ST_Intersection") {
    val data = Seq(
      Row(GeometryUDT.FromWkt("POINT (20 20)"), GeometryUDT.FromWkt("POLYGON ((0 0, 40 0, 40 40, 0 40, 0 0))")),
      // TODO: Add empty geometry support
      // Row(GeometryUDT.FromWkt("POINT (50 50)"), GeometryUDT.FromWkt("POLYGON ((0 0, 40 0, 40 40, 0 40, 0 0))")),
      Row(GeometryUDT.FromWkt("POLYGON ((10 10, 20 10, 20 20, 10 20, 10 10))"), GeometryUDT.FromWkt("POLYGON ((0 0, 40 0, 40 40, 0 40, 0 0))")),
      Row(GeometryUDT.FromWkt("POLYGON ((10 10, 50 10, 50 50, 10 50, 10 10))"), GeometryUDT.FromWkt("POLYGON ((0 0, 40 0, 40 40, 0 40, 0 0))"))
    )

    val schema = StructType(Array(StructField("left_geo", new GeometryUDT, nullable = true), StructField("right_geo", new GeometryUDT, nullable = true)))
    val df = spark.createDataFrame(spark.sparkContext.parallelize(data), schema)
    df.createOrReplaceTempView("data")

    val rst = spark.sql("select ST_Intersection(left_geo, right_geo) from data ")

    //    rst.queryExecution.debug.codegen()

    rst.show(false)
    val collect = rst.collect()

    val rst2 = df.select(st_intersection(col("left_geo"), col("right_geo")))
    rst2.show(false)
    val collect2 = rst2.collect()
  }

  test("ST_Intersection-Null") {
    val data = Seq(
      Row(null, "POLYGON ((0 0, 40 0, 40 40, 0 40, 0 0))"),
      Row("POINT (50 50)", null),
      Row(null, null),
      Row("POLYGON ((10 10, 50 10, 50 50, 10 50, 10 10))", "POLYGON ((0 0, 40 0, 40 40, 0 40, 0 0))")
    )

    val schema = StructType(Array(StructField("left_geo", StringType, nullable = true), StructField("right_geo", StringType, nullable = true)))
    val df = spark.createDataFrame(spark.sparkContext.parallelize(data), schema)
    df.createOrReplaceTempView("data")

    val rst = spark.sql("select ST_Intersection(ST_GeomFromText(left_geo), ST_GeomFromText(right_geo)) from data ")
    val collect = rst.collect()

    //    rst.queryExecution.debug.codegen()

    assert(collect(0).isNullAt(0))
    assert(collect(2).isNullAt(0))

    rst.show(false)

    val rst2 = df.select(st_intersection(st_geomfromtext(col("left_geo")), st_geomfromtext(col("right_geo"))))
    rst2.show(false)
    val collect2 = rst2.collect()
  }

  test("ST_SimplifyPreserveTopology") {
    val data = Seq(
      Row(GeometryUDT.FromWkt("Polygon((0 0, 0 1, 1 1, 1 0, 0 0))")),
      Row(GeometryUDT.FromWkt("LINESTRING (0 0, 10 10, 20 20)")),
      Row(GeometryUDT.FromWkt("POINT (0 0)")),
      Row(GeometryUDT.FromWkt("POLYGON EMPTY")),
      Row(GeometryUDT.FromWkt("MULTIPOLYGON ( ((0 0, 1 0, 1 1,0 0)) )"))
    )

    val schema = StructType(Array(StructField("geo", new GeometryUDT, nullable = true)))
    val df = spark.createDataFrame(spark.sparkContext.parallelize(data), schema)
    df.createOrReplaceTempView("data")

    val rst = spark.sql("select ST_SimplifyPreserveTopology(geo, 2) from data ")

    //    rst.queryExecution.debug.codegen()

    rst.show(false)
    val collect = rst.collect()

    val rst2 = df.select(st_simplifypreservetopology(col("geo"), lit(2)))
    rst2.show(false)
    val collect2 = rst2.collect()
  }

  test("ST_SimplifyPreserveTopology-Null") {
    val data = Seq(
      Row("error geo"),
      Row("LINESTRING (0 0, 10 10, 20 20)"),
      Row(null),
      Row("POLYGON ((1 2,3 4,5 6,1 2))"),
      Row("MULTIPOLYGON ( ((0 0, 1 0, 1 1,0 0)) )"),
    )

    val schema = StructType(Array(StructField("geo", StringType, nullable = true)))
    val df = spark.createDataFrame(spark.sparkContext.parallelize(data), schema)
    df.createOrReplaceTempView("data")

    val rst = spark.sql("select ST_SimplifyPreserveTopology(ST_GeomFromText(geo), 2) from data ")
    val collect = rst.collect()

    //    rst.queryExecution.debug.codegen()

    assert(collect(0).isNullAt(0))
    assert(collect(2).isNullAt(0))

    rst.show(false)

    val rst2 = df.select(st_simplifypreservetopology(st_geomfromtext(col("geo")), lit(2)))
    rst2.show(false)
    val collect2 = rst2.collect()
  }

  test("ST_ConvexHull") {
    val data = Seq(
      Row(GeometryUDT.FromWkt("Polygon((0 0, 0 1, 1 1, 1 0, 0 0))")),
      Row(GeometryUDT.FromWkt("LINESTRING (0 0, 10 10, 20 20)")),
      Row(GeometryUDT.FromWkt("POINT (0 0)")),
      Row(GeometryUDT.FromWkt("POLYGON EMPTY")),
      Row(GeometryUDT.FromWkt("MULTIPOLYGON ( ((0 0, 1 0, 1 1,0 0)) )"))
    )

    val schema = StructType(Array(StructField("geo", new GeometryUDT, nullable = true)))
    val df = spark.createDataFrame(spark.sparkContext.parallelize(data), schema)
    df.createOrReplaceTempView("data")

    val rst = spark.sql("select ST_ConvexHull(geo) from data ")

    //    rst.queryExecution.debug.codegen()

    rst.show(false)
    val collect = rst.collect()

    val rst2 = df.select(st_convexhull(col("geo")))
    rst2.show(false)
    val collect2 = rst2.collect()
  }

  test("ST_ConvexHull-Null") {
    val data = Seq(
      Row("error geo"),
      Row("LINESTRING (0 0, 10 10, 20 20)"),
      Row(null),
      Row("POLYGON ((1 2,3 4,5 6,1 2))"),
      Row("MULTIPOLYGON ( ((0 0, 1 0, 1 1,0 0)) )"),
    )

    val schema = StructType(Array(StructField("geo", StringType, nullable = true)))
    val df = spark.createDataFrame(spark.sparkContext.parallelize(data), schema)
    df.createOrReplaceTempView("data")

    val rst = spark.sql("select ST_ConvexHull(ST_GeomFromText(geo)) from data ")
    val collect = rst.collect()

    //    rst.queryExecution.debug.codegen()

    assert(collect(0).isNullAt(0))
    assert(collect(2).isNullAt(0))

    rst.show(false)

    val rst2 = df.select(st_convexhull(st_geomfromtext(col("geo"))))
    rst2.show(false)
    val collect2 = rst2.collect()
  }

  test("ST_Area") {
    val data = Seq(
      Row(GeometryUDT.FromWkt("Polygon((0 0, 0 1, 1 1, 1 0, 0 0))")),
      Row(GeometryUDT.FromWkt("LINESTRING (0 0, 10 10, 20 20)")),
      Row(GeometryUDT.FromWkt("POINT (0 0)")),
      Row(GeometryUDT.FromWkt("POLYGON EMPTY")),
      Row(GeometryUDT.FromWkt("MULTIPOLYGON ( ((0 0, 1 0, 1 1,0 0)) )"))
    )

    val schema = StructType(Array(StructField("geo", new GeometryUDT, nullable = true)))
    val df = spark.createDataFrame(spark.sparkContext.parallelize(data), schema)
    df.createOrReplaceTempView("data")

    val rst = spark.sql("select ST_Area(geo) from data ")

    //    rst.queryExecution.debug.codegen()

    rst.show(false)
    val collect = rst.collect()

    val rst2 = df.select(st_area(col("geo")))
    rst2.show(false)
    val collect2 = rst2.collect()
  }

  test("ST_Area-Null") {
    val data = Seq(
      Row("error geo"),
      Row("LINESTRING (0 0, 10 10, 20 20)"),
      Row(null),
      Row("POLYGON ((1 2,3 4,5 6,1 2))"),
      Row("MULTIPOLYGON ( ((0 0, 1 0, 1 1,0 0)) )"),
    )

    val schema = StructType(Array(StructField("geo", StringType, nullable = true)))
    val df = spark.createDataFrame(spark.sparkContext.parallelize(data), schema)
    df.createOrReplaceTempView("data")

    val rst = spark.sql("select ST_Area(ST_GeomFromText(geo)) from data ")
    val collect = rst.collect()

    //    rst.queryExecution.debug.codegen()

    assert(collect(0).isNullAt(0))
    assert(collect(2).isNullAt(0))

    rst.show(false)

    val rst2 = df.select(st_area(st_geomfromtext(col("geo"))))
    rst2.show(false)
    val collect2 = rst2.collect()
  }

  test("ST_Length") {
    val data = Seq(
      Row(GeometryUDT.FromWkt("Polygon((0 0, 0 1, 1 1, 1 0, 0 0))")),
      Row(GeometryUDT.FromWkt("LINESTRING (0 0, 10 10, 20 20)")),
      Row(GeometryUDT.FromWkt("POINT (0 0)")),
      Row(GeometryUDT.FromWkt("POLYGON EMPTY")),
      Row(GeometryUDT.FromWkt("MULTIPOLYGON ( ((0 0, 1 0, 1 1,0 0)) )"))
    )

    val schema = StructType(Array(StructField("geo", new GeometryUDT, nullable = true)))
    val df = spark.createDataFrame(spark.sparkContext.parallelize(data), schema)
    df.createOrReplaceTempView("data")

    val rst = spark.sql("select ST_Length(geo) from data ")

    //    rst.queryExecution.debug.codegen()

    rst.show(false)
    val collect = rst.collect()

    val rst2 = df.select(st_length(col("geo")))
    rst2.show(false)
    val collect2 = rst2.collect()
  }

  test("ST_Length-Null") {
    val data = Seq(
      Row("error geo"),
      Row("LINESTRING (0 0, 10 10, 20 20)"),
      Row(null),
      Row("POLYGON ((1 2,3 4,5 6,1 2))"),
      Row("MULTIPOLYGON ( ((0 0, 1 0, 1 1,0 0)) )"),
    )

    val schema = StructType(Array(StructField("geo", StringType, nullable = true)))
    val df = spark.createDataFrame(spark.sparkContext.parallelize(data), schema)
    df.createOrReplaceTempView("data")

    val rst = spark.sql("select ST_Length(ST_GeomFromText(geo)) from data ")
    val collect = rst.collect()

    //    rst.queryExecution.debug.codegen()

    assert(collect(0).isNullAt(0))
    assert(collect(2).isNullAt(0))

    rst.show(false)

    val rst2 = df.select(st_length(st_geomfromtext(col("geo"))))
    rst2.show(false)
    val collect2 = rst2.collect()
  }

  test("ST_HausdorffDistance") {
    val data = Seq(
      Row(GeometryUDT.FromWkt("POINT (20 20)"), GeometryUDT.FromWkt("POLYGON ((0 0, 40 0, 40 40, 0 40, 0 0))")),
      Row(GeometryUDT.FromWkt("POINT (50 50)"), GeometryUDT.FromWkt("POLYGON ((0 0, 40 0, 40 40, 0 40, 0 0))")),
      Row(GeometryUDT.FromWkt("POLYGON ((10 10, 20 10, 20 20, 10 20, 10 10))"), GeometryUDT.FromWkt("POLYGON ((0 0, 40 0, 40 40, 0 40, 0 0))")),
      Row(GeometryUDT.FromWkt("POLYGON ((10 10, 50 10, 50 50, 10 50, 10 10))"), GeometryUDT.FromWkt("POLYGON ((0 0, 40 0, 40 40, 0 40, 0 0))"))
    )

    val schema = StructType(Array(StructField("left_geo", new GeometryUDT, nullable = true), StructField("right_geo", new GeometryUDT, nullable = true)))
    val df = spark.createDataFrame(spark.sparkContext.parallelize(data), schema)
    df.createOrReplaceTempView("data")

    val rst = spark.sql("select ST_HausdorffDistance(left_geo, right_geo) from data ")

    //    rst.queryExecution.debug.codegen()

    rst.show(false)
    val collect = rst.collect()

    val rst2 = df.select(st_hausdorffdistance(col("left_geo"), col("right_geo")))
    rst2.show(false)
    val collect2 = rst2.collect()
  }

  test("ST_HausdorffDistance-Null") {
    val data = Seq(
      Row(null, "POLYGON ((0 0, 40 0, 40 40, 0 40, 0 0))"),
      Row("POINT (50 50)", null),
      Row(null, null),
      Row("POLYGON ((10 10, 50 10, 50 50, 10 50, 10 10))", "POLYGON ((0 0, 40 0, 40 40, 0 40, 0 0))")
    )

    val schema = StructType(Array(StructField("left_geo", StringType, nullable = true), StructField("right_geo", StringType, nullable = true)))
    val df = spark.createDataFrame(spark.sparkContext.parallelize(data), schema)
    df.createOrReplaceTempView("data")

    val rst = spark.sql("select ST_HausdorffDistance(ST_GeomFromText(left_geo), ST_GeomFromText(right_geo)) from data ")
    val collect = rst.collect()

    //    rst.queryExecution.debug.codegen()

    assert(collect(0).isNullAt(0))
    assert(collect(2).isNullAt(0))

    rst.show(false)

    val rst2 = df.select(st_hausdorffdistance(st_geomfromtext(col("left_geo")), st_geomfromtext(col("right_geo"))))
    rst2.show(false)
    val collect2 = rst2.collect()
  }

  test("ST_Distance") {
    val data = Seq(
      Row(GeometryUDT.FromWkt("POINT (20 20)"), GeometryUDT.FromWkt("POLYGON ((0 0, 40 0, 40 40, 0 40, 0 0))")),
      Row(GeometryUDT.FromWkt("POINT (50 50)"), GeometryUDT.FromWkt("POLYGON ((0 0, 40 0, 40 40, 0 40, 0 0))")),
      Row(GeometryUDT.FromWkt("POLYGON ((10 10, 20 10, 20 20, 10 20, 10 10))"), GeometryUDT.FromWkt("POLYGON ((0 0, 40 0, 40 40, 0 40, 0 0))")),
      Row(GeometryUDT.FromWkt("POLYGON ((10 10, 50 10, 50 50, 10 50, 10 10))"), GeometryUDT.FromWkt("POLYGON ((0 0, 40 0, 40 40, 0 40, 0 0))"))
    )

    val schema = StructType(Array(StructField("left_geo", new GeometryUDT, nullable = true), StructField("right_geo", new GeometryUDT, nullable = true)))
    val df = spark.createDataFrame(spark.sparkContext.parallelize(data), schema)
    df.createOrReplaceTempView("data")

    val rst = spark.sql("select ST_Distance(left_geo, right_geo) from data ")

    //    rst.queryExecution.debug.codegen()

    rst.show(false)
    val collect = rst.collect()

    val rst2 = df.select(st_distance(col("left_geo"), col("right_geo")))
    rst2.show(false)
    val collect2 = rst2.collect()
  }

  test("ST_Distance-Null") {
    val data = Seq(
      Row(null, "POLYGON ((0 0, 40 0, 40 40, 0 40, 0 0))"),
      Row("POINT (50 50)", null),
      Row(null, null),
      Row("POLYGON ((10 10, 50 10, 50 50, 10 50, 10 10))", "POLYGON ((0 0, 40 0, 40 40, 0 40, 0 0))")
    )

    val schema = StructType(Array(StructField("left_geo", StringType, nullable = true), StructField("right_geo", StringType, nullable = true)))
    val df = spark.createDataFrame(spark.sparkContext.parallelize(data), schema)
    df.createOrReplaceTempView("data")

    val rst = spark.sql("select ST_Distance(ST_GeomFromText(left_geo), ST_GeomFromText(right_geo)) from data ")
    val collect = rst.collect()

    //    rst.queryExecution.debug.codegen()

    assert(collect(0).isNullAt(0))
    assert(collect(2).isNullAt(0))

    rst.show(false)

    val rst2 = df.select(st_distance(st_geomfromtext(col("left_geo")), st_geomfromtext(col("right_geo"))))
    rst2.show(false)
    val collect2 = rst2.collect()
  }

  test("ST_Equals") {
    val data = Seq(
      Row(GeometryUDT.FromWkt("POINT (20 20)"), GeometryUDT.FromWkt("POINT (20 20)")),
      Row(GeometryUDT.FromWkt("POINT (50 50)"), GeometryUDT.FromWkt("POLYGON ((0 0, 40 0, 40 40, 0 40, 0 0))")),
      Row(GeometryUDT.FromWkt("POLYGON ((10 10, 20 10, 20 20, 10 20, 10 10))"), GeometryUDT.FromWkt("POLYGON ((0 0, 40 0, 40 40, 0 40, 0 0))")),
      Row(GeometryUDT.FromWkt("POLYGON ((10 10, 50 10, 50 50, 10 50, 10 10))"), GeometryUDT.FromWkt("POLYGON ((10 10, 50 10, 50 50, 10 50, 10 10))"))
    )

    val schema = StructType(Array(StructField("left_geo", new GeometryUDT, nullable = true), StructField("right_geo", new GeometryUDT, nullable = true)))
    val df = spark.createDataFrame(spark.sparkContext.parallelize(data), schema)
    df.createOrReplaceTempView("data")

    val rst = spark.sql("select ST_Equals(left_geo, right_geo) from data ")

    //    rst.queryExecution.debug.codegen()

    rst.show(false)
    val collect = rst.collect()

    val rst2 = df.select(st_equals(col("left_geo"), col("right_geo")))
    rst2.show(false)
    val collect2 = rst2.collect()
  }

  test("ST_Equals-Null") {
    val data = Seq(
      Row(null, "POLYGON ((0 0, 40 0, 40 40, 0 40, 0 0))"),
      Row("POINT (50 50)", null),
      Row(null, null),
      Row("POLYGON ((10 10, 50 10, 50 50, 10 50, 10 10))", "POLYGON ((10 10, 50 10, 50 50, 10 50, 10 10))")
    )

    val schema = StructType(Array(StructField("left_geo", StringType, nullable = true), StructField("right_geo", StringType, nullable = true)))
    val df = spark.createDataFrame(spark.sparkContext.parallelize(data), schema)
    df.createOrReplaceTempView("data")

    val rst = spark.sql("select ST_Equals(ST_GeomFromText(left_geo), ST_GeomFromText(right_geo)) from data ")
    val collect = rst.collect()

    //    rst.queryExecution.debug.codegen()

    assert(collect(0).isNullAt(0))
    assert(collect(2).isNullAt(0))

    rst.show(false)

    val rst2 = df.select(st_equals(st_geomfromtext(col("left_geo")), st_geomfromtext(col("right_geo"))))
    rst2.show(false)
    val collect2 = rst2.collect()
  }

  test("ST_Touches") {
    val data = Seq(
      Row(GeometryUDT.FromWkt("POINT (0 0)"), GeometryUDT.FromWkt("POLYGON ((0 0, 40 0, 40 40, 0 40, 0 0))")),
      Row(GeometryUDT.FromWkt("POINT (50 50)"), GeometryUDT.FromWkt("POLYGON ((0 0, 40 0, 40 40, 0 40, 0 0))")),
      Row(GeometryUDT.FromWkt("POLYGON ((40 40, 80 40, 80 80, 40 80, 40 40))"), GeometryUDT.FromWkt("POLYGON ((0 0, 40 0, 40 40, 0 40, 0 0))")),
      Row(GeometryUDT.FromWkt("POLYGON ((10 10, 50 10, 50 50, 10 50, 10 10))"), GeometryUDT.FromWkt("POLYGON ((0 0, 40 0, 40 40, 0 40, 0 0))"))
    )

    val schema = StructType(Array(StructField("left_geo", new GeometryUDT, nullable = true), StructField("right_geo", new GeometryUDT, nullable = true)))
    val df = spark.createDataFrame(spark.sparkContext.parallelize(data), schema)
    df.createOrReplaceTempView("data")

    val rst = spark.sql("select ST_Touches(left_geo, right_geo) from data ")

    //    rst.queryExecution.debug.codegen()

    rst.show(false)
    val collect = rst.collect()

    val rst2 = df.select(st_touches(col("left_geo"), col("right_geo")))
    rst2.show(false)
    val collect2 = rst2.collect()
  }

  test("ST_Touches-Null") {
    val data = Seq(
      Row(null, "POLYGON ((0 0, 40 0, 40 40, 0 40, 0 0))"),
      Row("POINT (50 50)", null),
      Row(null, null),
      Row("POLYGON ((10 10, 50 10, 50 50, 10 50, 10 10))", "POLYGON ((0 0, 40 0, 40 40, 0 40, 0 0))")
    )

    val schema = StructType(Array(StructField("left_geo", StringType, nullable = true), StructField("right_geo", StringType, nullable = true)))
    val df = spark.createDataFrame(spark.sparkContext.parallelize(data), schema)
    df.createOrReplaceTempView("data")

    val rst = spark.sql("select ST_Touches(ST_GeomFromText(left_geo), ST_GeomFromText(right_geo)) from data ")
    val collect = rst.collect()

    //    rst.queryExecution.debug.codegen()

    assert(collect(0).isNullAt(0))
    assert(collect(2).isNullAt(0))

    rst.show(false)

    val rst2 = df.select(st_touches(st_geomfromtext(col("left_geo")), st_geomfromtext(col("right_geo"))))
    rst2.show(false)
    val collect2 = rst2.collect()
  }

  test("ST_Overlaps") {
    val data = Seq(
      Row(GeometryUDT.FromWkt("POINT (0 0)"), GeometryUDT.FromWkt("POLYGON ((0 0, 40 0, 40 40, 0 40, 0 0))")),
      Row(GeometryUDT.FromWkt("POINT (50 50)"), GeometryUDT.FromWkt("POLYGON ((0 0, 40 0, 40 40, 0 40, 0 0))")),
      Row(GeometryUDT.FromWkt("POLYGON ((40 40, 80 40, 80 80, 40 80, 40 40))"), GeometryUDT.FromWkt("POLYGON ((0 0, 40 0, 40 40, 0 40, 0 0))")),
      Row(GeometryUDT.FromWkt("POLYGON ((10 10, 50 10, 50 50, 10 50, 10 10))"), GeometryUDT.FromWkt("POLYGON ((0 0, 40 0, 40 40, 0 40, 0 0))"))
    )

    val schema = StructType(Array(StructField("left_geo", new GeometryUDT, nullable = true), StructField("right_geo", new GeometryUDT, nullable = true)))
    val df = spark.createDataFrame(spark.sparkContext.parallelize(data), schema)
    df.createOrReplaceTempView("data")

    val rst = spark.sql("select ST_Overlaps(left_geo, right_geo) from data ")

    //    rst.queryExecution.debug.codegen()

    rst.show(false)
    val collect = rst.collect()

    val rst2 = df.select(st_overlaps(col("left_geo"), col("right_geo")))
    rst2.show(false)
    val collect2 = rst2.collect()
  }

  test("ST_Overlaps-Null") {
    val data = Seq(
      Row(null, "POLYGON ((0 0, 40 0, 40 40, 0 40, 0 0))"),
      Row("POINT (50 50)", null),
      Row(null, null),
      Row("POLYGON ((10 10, 50 10, 50 50, 10 50, 10 10))", "POLYGON ((0 0, 40 0, 40 40, 0 40, 0 0))")
    )

    val schema = StructType(Array(StructField("left_geo", StringType, nullable = true), StructField("right_geo", StringType, nullable = true)))
    val df = spark.createDataFrame(spark.sparkContext.parallelize(data), schema)
    df.createOrReplaceTempView("data")

    val rst = spark.sql("select ST_Overlaps(ST_GeomFromText(left_geo), ST_GeomFromText(right_geo)) from data ")
    val collect = rst.collect()

    //    rst.queryExecution.debug.codegen()

    assert(collect(0).isNullAt(0))
    assert(collect(2).isNullAt(0))

    rst.show(false)

    val rst2 = df.select(st_overlaps(st_geomfromtext(col("left_geo")), st_geomfromtext(col("right_geo"))))
    rst2.show(false)
    val collect2 = rst2.collect()
  }

  test("ST_Crosses") {
    val data = Seq(
      Row(GeometryUDT.FromWkt("LINESTRING (0 0, 80 80)"), GeometryUDT.FromWkt("POLYGON ((0 0, 40 0, 40 40, 0 40, 0 0))")),
      Row(GeometryUDT.FromWkt("POINT (50 50)"), GeometryUDT.FromWkt("POLYGON ((0 0, 40 0, 40 40, 0 40, 0 0))")),
      Row(GeometryUDT.FromWkt("POLYGON ((40 40, 80 40, 80 80, 40 80, 40 40))"), GeometryUDT.FromWkt("POLYGON ((0 0, 40 0, 40 40, 0 40, 0 0))")),
      Row(GeometryUDT.FromWkt("POLYGON ((10 10, 50 10, 50 50, 10 50, 10 10))"), GeometryUDT.FromWkt("POLYGON ((0 0, 40 0, 40 40, 0 40, 0 0))"))
    )

    val schema = StructType(Array(StructField("left_geo", new GeometryUDT, nullable = true), StructField("right_geo", new GeometryUDT, nullable = true)))
    val df = spark.createDataFrame(spark.sparkContext.parallelize(data), schema)
    df.createOrReplaceTempView("data")

    val rst = spark.sql("select ST_Crosses(left_geo, right_geo) from data ")

    //    rst.queryExecution.debug.codegen()

    rst.show(false)
    val collect = rst.collect()

    val rst2 = df.select(st_crosses(col("left_geo"), col("right_geo")))
    rst2.show(false)
    val collect2 = rst2.collect()
  }

  test("ST_Crosses-Null") {
    val data = Seq(
      Row(null, "POLYGON ((0 0, 40 0, 40 40, 0 40, 0 0))"),
      Row("POINT (50 50)", null),
      Row(null, null),
      Row("POLYGON ((10 10, 50 10, 50 50, 10 50, 10 10))", "POLYGON ((0 0, 40 0, 40 40, 0 40, 0 0))")
    )

    val schema = StructType(Array(StructField("left_geo", StringType, nullable = true), StructField("right_geo", StringType, nullable = true)))
    val df = spark.createDataFrame(spark.sparkContext.parallelize(data), schema)
    df.createOrReplaceTempView("data")

    val rst = spark.sql("select ST_Crosses(ST_GeomFromText(left_geo), ST_GeomFromText(right_geo)) from data ")
    val collect = rst.collect()

    //    rst.queryExecution.debug.codegen()

    assert(collect(0).isNullAt(0))
    assert(collect(2).isNullAt(0))

    rst.show(false)

    val rst2 = df.select(st_crosses(st_geomfromtext(col("left_geo")), st_geomfromtext(col("right_geo"))))
    rst2.show(false)
    val collect2 = rst2.collect()
  }

  test("ST_Contains") {
    val data = Seq(
      Row(GeometryUDT.FromWkt("POINT (20 20)"), GeometryUDT.FromWkt("POLYGON ((0 0, 40 0, 40 40, 0 40, 0 0))")),
      Row(GeometryUDT.FromWkt("POLYGON ((0 0, 40 0, 40 40, 0 40, 0 0))"), GeometryUDT.FromWkt("POINT (20 20)")),
      Row(GeometryUDT.FromWkt("POLYGON ((40 40, 80 40, 80 80, 40 80, 40 40))"), GeometryUDT.FromWkt("POLYGON ((0 0, 40 0, 40 40, 0 40, 0 0))")),
      Row(GeometryUDT.FromWkt("POLYGON ((0 0, 40 0, 40 40, 0 40, 0 0))"), GeometryUDT.FromWkt("POLYGON ((10 10, 20 10, 20 20, 10 20, 10 10))"))
    )

    val schema = StructType(Array(StructField("left_geo", new GeometryUDT, nullable = true), StructField("right_geo", new GeometryUDT, nullable = true)))
    val df = spark.createDataFrame(spark.sparkContext.parallelize(data), schema)
    df.createOrReplaceTempView("data")

    val rst = spark.sql("select ST_Contains(left_geo, right_geo) from data ")

    //    rst.queryExecution.debug.codegen()

    rst.show(false)
    val collect = rst.collect()

    val rst2 = df.select(st_contains(col("left_geo"), col("right_geo")))
    rst2.show(false)
    val collect2 = rst2.collect()
  }

  test("ST_Contains-Null") {
    val data = Seq(
      Row(null, "POLYGON ((0 0, 40 0, 40 40, 0 40, 0 0))"),
      Row("POINT (50 50)", null),
      Row(null, null),
      Row("POLYGON ((10 10, 50 10, 50 50, 10 50, 10 10))", "POLYGON ((0 0, 40 0, 40 40, 0 40, 0 0))")
    )

    val schema = StructType(Array(StructField("left_geo", StringType, nullable = true), StructField("right_geo", StringType, nullable = true)))
    val df = spark.createDataFrame(spark.sparkContext.parallelize(data), schema)
    df.createOrReplaceTempView("data")

    val rst = spark.sql("select ST_Contains(ST_GeomFromText(left_geo), ST_GeomFromText(right_geo)) from data ")
    val collect = rst.collect()

    //    rst.queryExecution.debug.codegen()

    assert(collect(0).isNullAt(0))
    assert(collect(2).isNullAt(0))

    rst.show(false)

    val rst2 = df.select(st_contains(st_geomfromtext(col("left_geo")), st_geomfromtext(col("right_geo"))))
    rst2.show(false)
    val collect2 = rst2.collect()
  }

  test("ST_Intersects") {
    val data = Seq(
      Row(GeometryUDT.FromWkt("POINT (20 20)"), GeometryUDT.FromWkt("POLYGON ((0 0, 40 0, 40 40, 0 40, 0 0))")),
      Row(GeometryUDT.FromWkt("POLYGON ((0 0, 40 0, 40 40, 0 40, 0 0))"), GeometryUDT.FromWkt("POINT (50 50)")),
      Row(GeometryUDT.FromWkt("POLYGON ((40 40, 80 40, 80 80, 40 80, 40 40))"), GeometryUDT.FromWkt("POLYGON ((0 0, 40 0, 40 40, 0 40, 0 0))")),
      Row(GeometryUDT.FromWkt("POLYGON ((0 0, 40 0, 40 40, 0 40, 0 0))"), GeometryUDT.FromWkt("POLYGON ((10 10, 20 10, 20 20, 10 20, 10 10))"))
    )

    val schema = StructType(Array(StructField("left_geo", new GeometryUDT, nullable = true), StructField("right_geo", new GeometryUDT, nullable = true)))
    val df = spark.createDataFrame(spark.sparkContext.parallelize(data), schema)
    df.createOrReplaceTempView("data")

    val rst = spark.sql("select ST_Intersects(left_geo, right_geo) from data ")

    //    rst.queryExecution.debug.codegen()

    rst.show(false)
    val collect = rst.collect()

    val rst2 = df.select(st_intersects(col("left_geo"), col("right_geo")))
    rst2.show(false)
    val collect2 = rst2.collect()
  }

  test("ST_Intersects-Null") {
    val data = Seq(
      Row(null, "POLYGON ((0 0, 40 0, 40 40, 0 40, 0 0))"),
      Row("POINT (50 50)", null),
      Row(null, null),
      Row("POLYGON ((10 10, 50 10, 50 50, 10 50, 10 10))", "POLYGON ((0 0, 40 0, 40 40, 0 40, 0 0))")
    )

    val schema = StructType(Array(StructField("left_geo", StringType, nullable = true), StructField("right_geo", StringType, nullable = true)))
    val df = spark.createDataFrame(spark.sparkContext.parallelize(data), schema)
    df.createOrReplaceTempView("data")

    val rst = spark.sql("select ST_Intersects(ST_GeomFromText(left_geo), ST_GeomFromText(right_geo)) from data ")
    val collect = rst.collect()

    //    rst.queryExecution.debug.codegen()

    assert(collect(0).isNullAt(0))
    assert(collect(2).isNullAt(0))

    rst.show(false)

    val rst2 = df.select(st_intersects(st_geomfromtext(col("left_geo")), st_geomfromtext(col("right_geo"))))
    rst2.show(false)
    val collect2 = rst2.collect()
  }

  test("ST_DistanceSphere") {
    val data = Seq(
      Row(GeometryUDT.FromWkt("POINT (-73.981153 40.741841)"), GeometryUDT.FromWkt("POINT (-73.990167 40.729884)")),
      Row(GeometryUDT.FromWkt("POINT (-74.123512 40.561438)"), GeometryUDT.FromWkt("POINT (-73.418598 41.681739)")),
    )

    val schema = StructType(Array(StructField("left_geo", new GeometryUDT, nullable = true), StructField("right_geo", new GeometryUDT, nullable = true)))
    val df = spark.createDataFrame(spark.sparkContext.parallelize(data), schema)
    df.createOrReplaceTempView("data")

    val rst = spark.sql("select ST_DistanceSphere(left_geo, right_geo) from data ")

    //    rst.queryExecution.debug.codegen()

    rst.show(false)
    val collect = rst.collect()

    val rst2 = df.select(st_distancesphere(col("left_geo"), col("right_geo")))
    rst2.show(false)
    val collect2 = rst2.collect()
  }

  test("ST_DistanceSphere-Null") {
    val data = Seq(
      Row(null, "POINT (-73.981153 40.741841)"),
      Row("POINT (-73.990167 40.729884)", null),
      Row(null, null),
      Row("POINT (-73.981153 40.741841)", "POLYGON ((0 0, 40 0, 40 40, 0 40, 0 0))"), // illegal geometry
      Row("POINT (-200 40.741841)", "POINT (-73.981153 40.741841)"), // illegal coordinate
    )

    val schema = StructType(Array(StructField("left_geo", StringType, nullable = true), StructField("right_geo", StringType, nullable = true)))
    val df = spark.createDataFrame(spark.sparkContext.parallelize(data), schema)
    df.createOrReplaceTempView("data")

    val rst = spark.sql("select ST_DistanceSphere(ST_GeomFromText(left_geo), ST_GeomFromText(right_geo)) from data ")
    val collect = rst.collect()

    //    rst.queryExecution.debug.codegen()

    assert(collect(0).isNullAt(0))
    assert(collect(1).isNullAt(0))
    assert(collect(2).isNullAt(0))

    rst.show(false)

    val rst2 = df.select(st_distancesphere(st_geomfromtext(col("left_geo")), st_geomfromtext(col("right_geo"))))
    rst2.show(false)
    val collect2 = rst2.collect()
  }

  test("ST_Transform") {
    val data = Seq(
      Row(GeometryUDT.FromWkt("POINT (-73.981153 40.741841)")),
      Row(GeometryUDT.FromWkt("POINT (-74.123512 40.561438)")),
    )

    val schema = StructType(Array(StructField("geo", new GeometryUDT, nullable = true)))
    val df = spark.createDataFrame(spark.sparkContext.parallelize(data), schema)
    df.createOrReplaceTempView("data")

    val rst = spark.sql("select ST_Transform(geo, 'EPSG:4326', 'EPSG:3857') from data ")

    //    rst.queryExecution.debug.codegen()

    rst.show(false)
    val collect = rst.collect()

    val rst2 = df.select(st_transform(col("geo"), lit("EPSG:4326"), lit("EPSG:3857")))
    rst2.show(false)
    val collect2 = rst2.collect()
  }

  test("ST_Transform-Null") {
    val data = Seq(
      Row(null),
      Row("POINT (-73.990167 40.729884)"),
      Row("POLYGON ((0 0, 40 0, 40 40, 0 40, 0 0))"),
      Row("POINT (-73.981153 40.741841)"),
    )

    val schema = StructType(Array(StructField("geo", StringType, nullable = true)))
    val df = spark.createDataFrame(spark.sparkContext.parallelize(data), schema)
    df.createOrReplaceTempView("data")

    val rst = spark.sql("select ST_Transform(ST_GeomFromText(geo), 'EPSG:4326', 'EPSG:3857') from data ")
    val collect = rst.collect()

    //    rst.queryExecution.debug.codegen()

    assert(collect(0).isNullAt(0))

    rst.show(false)

    val rst2 = df.select(st_transform(st_geomfromtext(col("geo")), lit("EPSG:4326"), lit("EPSG:3857")))
    rst2.show(false)
    val collect2 = rst2.collect()
  }

  test("ST_MakeValid") {
    val data = Seq(
      Row(GeometryUDT.FromWkt("POLYGON ((0 0,0 1,1 2,0 0))")),
      Row(GeometryUDT.FromWkt("MULTIPOLYGON ( ((0 0, 0 4, 4 4, 4 0, 0 0)), ((0 0, 4 0, 4 1, 0 1, 0 0)) )")),
    )

    val schema = StructType(Array(StructField("geo", new GeometryUDT, nullable = true)))
    val df = spark.createDataFrame(spark.sparkContext.parallelize(data), schema)
    df.createOrReplaceTempView("data")

    val rst = spark.sql("select ST_MakeValid(geo) from data ")

    //    rst.queryExecution.debug.codegen()

    rst.show(false)
    val collect = rst.collect()

    val rst2 = df.select(st_makevalid(col("geo")))
    rst2.show(false)
    val collect2 = rst2.collect()
  }

  test("ST_MakeValid-Null") {
    val data = Seq(
      Row(null),
      Row("POINT (-73.990167 40.729884)"),
      Row("POLYGON ((0 0, 40 0, 40 40, 0 40, 0 0))"),
      Row("MULTIPOLYGON ( ((0 0, 0 4, 4 4, 4 0, 0 0)), ((0 0, 4 0, 4 1, 0 1, 0 0)) )"),
    )

    val schema = StructType(Array(StructField("geo", StringType, nullable = true)))
    val df = spark.createDataFrame(spark.sparkContext.parallelize(data), schema)
    df.createOrReplaceTempView("data")

    val rst = spark.sql("select ST_MakeValid(ST_GeomFromText(geo)) from data ")
    val collect = rst.collect()

    //    rst.queryExecution.debug.codegen()

    assert(collect(0).isNullAt(0))

    rst.show(false)

    val rst2 = df.select(st_makevalid(st_geomfromtext(col("geo"))))
    rst2.show(false)
    val collect2 = rst2.collect()
  }
}
