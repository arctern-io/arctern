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
import org.apache.spark.sql.arctern._
import org.apache.spark.sql.arctern.functions._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._

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
    rst.show(false)

    //    rst.queryExecution.debug.codegen()

    val collect = rst.collect()

    assert(collect(0).getBoolean(1) == true)
    assert(collect(1).getBoolean(1) == false)
    assert(collect(2).getBoolean(1) == true)
    assert(collect(3).getBoolean(1) == false)

    val rst2 = df.select(st_within(col("geo1"), col("geo2")))
    rst2.show(false)

    val collect2 = rst2.collect()

    assert(collect2(0).getBoolean(0) == true)
    assert(collect2(1).getBoolean(0) == false)
    assert(collect2(2).getBoolean(0) == true)
    assert(collect2(3).getBoolean(0) == false)
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
    rst.show(false)

    //    rst.queryExecution.debug.codegen()

    val collect = rst.collect()

    assert(collect(0).isNullAt(1))
    assert(collect(1).isNullAt(1))
    assert(collect(2).isNullAt(1))
    assert(collect(3).getBoolean(1) == false)

    val rst2 = df.select(st_within(st_geomfromtext(col("geo1")), st_geomfromtext(col("geo2"))))
    rst2.show(false)

    val collect2 = rst2.collect()

    assert(collect2(0).isNullAt(0))
    assert(collect2(1).isNullAt(0))
    assert(collect2(2).isNullAt(0))
    assert(collect2(3).getBoolean(0) == false)
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
    rst.show(false)

    //    rst.queryExecution.debug.codegen()

    val collect = rst.collect()

    assert(collect(0).getAs[GeometryUDT](1).toString == "POINT (0.5 0.5)")
    assert(collect(1).getAs[GeometryUDT](1).toString == "POINT (10 10)")

    val rst2 = df.select(st_centroid(col("geo")))
    rst2.show(false)

    val collect2 = rst2.collect()

    assert(collect2(0).getAs[GeometryUDT](0).toString == "POINT (0.5 0.5)")
    assert(collect2(1).getAs[GeometryUDT](0).toString == "POINT (10 10)")
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
    rst.show(false)

    //    rst.queryExecution.debug.codegen()

    val collect = rst.collect()

    assert(collect(0).isNullAt(1))
    assert(collect(1).getAs[GeometryUDT](1).toString == "POINT (10 10)")
    assert(collect(2).isNullAt(1))

    val rst2 = df.select(st_centroid(st_geomfromtext(col("geo"))))
    rst2.show(false)

    val collect2 = rst2.collect()

    assert(collect2(0).isNullAt(0))
    assert(collect2(1).getAs[GeometryUDT](0).toString == "POINT (10 10)")
    assert(collect2(2).isNullAt(0))
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
    rst.show(false)

    //    rst.queryExecution.debug.codegen()

    val collect = rst.collect()

    assert(collect(0).getBoolean(1) == true)
    assert(collect(1).isNullAt(1))
    assert(collect(2).isNullAt(1))

    val rst2 = df.select(st_within(st_centroid(st_geomfromtext(col("geo1"))), st_centroid(st_geomfromtext(col("geo2")))))
    rst2.show(false)

    val collect2 = rst2.collect()

    assert(collect2(0).getBoolean(0) == true)
    assert(collect2(1).isNullAt(0))
    assert(collect2(2).isNullAt(0))
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
    rst.show(false)

    //    rst.queryExecution.debug.codegen()

    val collect = rst.collect()

    assert(collect(0).getBoolean(0) == true)
    assert(collect(1).getBoolean(0) == true)
    assert(collect(2).getBoolean(0) == true)

    val rst2 = df.select(st_isvalid(col("geo")))
    rst2.show(false)

    val collect2 = rst2.collect()

    assert(collect2(0).getBoolean(0) == true)
    assert(collect2(1).getBoolean(0) == true)
    assert(collect2(2).getBoolean(0) == true)
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
    rst.show(false)

    //    rst.queryExecution.debug.codegen()

    val collect = rst.collect()

    assert(collect(0).isNullAt(0))
    assert(collect(1).getBoolean(0) == true)
    assert(collect(2).isNullAt(0))

    val rst2 = df.select(st_isvalid(st_geomfromtext(col("geo"))))
    rst2.show(false)

    val collect2 = rst2.collect()

    assert(collect2(0).isNullAt(0))
    assert(collect2(1).getBoolean(0) == true)
    assert(collect2(2).isNullAt(0))
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
    rst.show(false)

    //    rst.queryExecution.debug.codegen()

    val collect = rst.collect()

    assert(collect(0).getString(0) == "Polygon")
    assert(collect(1).getString(0) == "LineString")
    assert(collect(2).getString(0) == "Point")
    assert(collect(3).getString(0) == "Polygon")
    assert(collect(4).getString(0) == "MultiPolygon")

    val rst2 = df.select(st_geometrytype(col("geo")))
    rst2.show(false)

    val collect2 = rst2.collect()

    assert(collect2(0).getString(0) == "Polygon")
    assert(collect2(1).getString(0) == "LineString")
    assert(collect2(2).getString(0) == "Point")
    assert(collect2(3).getString(0) == "Polygon")
    assert(collect2(4).getString(0) == "MultiPolygon")
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
    rst.show(false)

    //    rst.queryExecution.debug.codegen()

    val collect = rst.collect()

    assert(collect(0).isNullAt(0))
    assert(collect(1).getString(0) == "LineString")
    assert(collect(2).isNullAt(0))
    assert(collect(3).getString(0) == "Polygon")
    assert(collect(4).getString(0) == "MultiPolygon")

    val rst2 = df.select(st_geometrytype(st_geomfromtext(col("geo"))))
    rst2.show(false)

    val collect2 = rst2.collect()

    assert(collect2(0).isNullAt(0))
    assert(collect2(1).getString(0) == "LineString")
    assert(collect2(2).isNullAt(0))
    assert(collect2(3).getString(0) == "Polygon")
    assert(collect2(4).getString(0) == "MultiPolygon")
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
    rst.show(false)

    //    rst.queryExecution.debug.codegen()

    val collect = rst.collect()

    assert(collect(0).getBoolean(0) == true)
    assert(collect(1).getBoolean(0) == true)
    assert(collect(2).getBoolean(0) == true)
    assert(collect(3).getBoolean(0) == false)
    assert(collect(4).getBoolean(0) == true)

    val rst2 = df.select(st_issimple(col("geo")))
    rst2.show(false)

    val collect2 = rst2.collect()

    assert(collect2(0).getBoolean(0) == true)
    assert(collect2(1).getBoolean(0) == true)
    assert(collect2(2).getBoolean(0) == true)
    assert(collect2(3).getBoolean(0) == false)
    assert(collect2(4).getBoolean(0) == true)
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
    rst.show(false)

    //    rst.queryExecution.debug.codegen()

    val collect = rst.collect()

    assert(collect(0).isNullAt(0))
    assert(collect(1).getBoolean(0) == true)
    assert(collect(2).isNullAt(0))
    assert(collect(3).getBoolean(0) == false)
    assert(collect(4).getBoolean(0) == true)

    val rst2 = df.select(st_issimple(st_geomfromtext(col("geo"))))
    rst2.show(false)

    val collect2 = rst2.collect()

    assert(collect2(0).isNullAt(0))
    assert(collect2(1).getBoolean(0) == true)
    assert(collect2(2).isNullAt(0))
    assert(collect2(3).getBoolean(0) == false)
    assert(collect2(4).getBoolean(0) == true)
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
    rst.show(false)

    //    rst.queryExecution.debug.codegen()

    val collect = rst.collect()

    assert(collect(0).getInt(0) == 5)
    assert(collect(1).getInt(0) == 3)
    assert(collect(2).getInt(0) == 1)
    assert(collect(3).getInt(0) == 0)
    assert(collect(4).getInt(0) == 4)

    val rst2 = df.select(st_npoints(col("geo")))
    rst2.show(false)

    val collect2 = rst2.collect()

    assert(collect2(0).getInt(0) == 5)
    assert(collect2(1).getInt(0) == 3)
    assert(collect2(2).getInt(0) == 1)
    assert(collect2(3).getInt(0) == 0)
    assert(collect2(4).getInt(0) == 4)
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
    rst.show(false)

    //    rst.queryExecution.debug.codegen()

    val collect = rst.collect()

    assert(collect(0).isNullAt(0))
    assert(collect(1).getInt(0) == 3)
    assert(collect(2).isNullAt(0))
    assert(collect(3).getInt(0) == 4)
    assert(collect(4).getInt(0) == 4)

    val rst2 = df.select(st_npoints(st_geomfromtext(col("geo"))))
    rst2.show(false)

    val collect2 = rst2.collect()

    assert(collect2(0).isNullAt(0))
    assert(collect2(1).getInt(0) == 3)
    assert(collect2(2).isNullAt(0))
    assert(collect2(3).getInt(0) == 4)
    assert(collect2(4).getInt(0) == 4)
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
    rst.show(false)

    //    rst.queryExecution.debug.codegen()

    val collect = rst.collect()

    assert(collect(0).getAs[GeometryUDT](0).toString == "POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))")
    assert(collect(1).getAs[GeometryUDT](0).toString == "POLYGON ((0 0, 0 20, 20 20, 20 0, 0 0))")
    assert(collect(2).getAs[GeometryUDT](0).toString == "POINT (0 0)")
    assert(collect(3).getAs[GeometryUDT](0).toString == "POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))")

    val rst2 = df.select(st_envelope(col("geo")))
    rst2.show(false)

    val collect2 = rst2.collect()

    assert(collect2(0).getAs[GeometryUDT](0).toString == "POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))")
    assert(collect2(1).getAs[GeometryUDT](0).toString == "POLYGON ((0 0, 0 20, 20 20, 20 0, 0 0))")
    assert(collect2(2).getAs[GeometryUDT](0).toString == "POINT (0 0)")
    assert(collect2(3).getAs[GeometryUDT](0).toString == "POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))")
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
    rst.show(false)

    //    rst.queryExecution.debug.codegen()

    val collect = rst.collect()

    assert(collect(0).isNullAt(0))
    assert(collect(1).getAs[GeometryUDT](0).toString == "POLYGON ((0 0, 0 20, 20 20, 20 0, 0 0))")
    assert(collect(2).isNullAt(0))
    assert(collect(3).getAs[GeometryUDT](0).toString == "POLYGON ((1 2, 1 6, 5 6, 5 2, 1 2))")
    assert(collect(4).getAs[GeometryUDT](0).toString == "POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))")

    val rst2 = df.select(st_envelope(st_geomfromtext(col("geo"))))
    rst2.show(false)

    val collect2 = rst2.collect()

    assert(collect2(0).isNullAt(0))
    assert(collect2(1).getAs[GeometryUDT](0).toString == "POLYGON ((0 0, 0 20, 20 20, 20 0, 0 0))")
    assert(collect2(2).isNullAt(0))
    assert(collect2(3).getAs[GeometryUDT](0).toString == "POLYGON ((1 2, 1 6, 5 6, 5 2, 1 2))")
    assert(collect2(4).getAs[GeometryUDT](0).toString == "POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))")
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

    val rst = spark.sql("select ST_Buffer(geo, 1 * 1) from data ")
    rst.show(false)

    //    rst.queryExecution.debug.codegen()

    val collect = rst.collect()

    assert(collect(0).getAs[GeometryUDT](0).toString == "POLYGON ((0 -1, -0.1950903220161287 -0.9807852804032303, -0.3826834323650903 -0.9238795325112865, -0.5555702330196022 -0.8314696123025452, -0.7071067811865477 -0.7071067811865475, -0.8314696123025455 -0.555570233019602, -0.9238795325112868 -0.3826834323650897, -0.9807852804032304 -0.1950903220161284, -1 0, -1 1, -0.9807852804032304 1.1950903220161286, -0.9238795325112867 1.3826834323650898, -0.8314696123025453 1.5555702330196022, -0.7071067811865475 1.7071067811865475, -0.555570233019602 1.8314696123025453, -0.3826834323650897 1.9238795325112867, -0.1950903220161282 1.9807852804032304, 0 2, 1 2, 1.1950903220161284 1.9807852804032304, 1.3826834323650898 1.9238795325112867, 1.5555702330196022 1.8314696123025453, 1.7071067811865475 1.7071067811865475, 1.8314696123025453 1.5555702330196022, 1.9238795325112867 1.3826834323650898, 1.9807852804032304 1.1950903220161282, 2 1, 2 0, 1.9807852804032304 -0.1950903220161282, 1.9238795325112867 -0.3826834323650898, 1.8314696123025453 -0.5555702330196022, 1.7071067811865475 -0.7071067811865475, 1.5555702330196022 -0.8314696123025452, 1.3826834323650898 -0.9238795325112867, 1.1950903220161284 -0.9807852804032304, 1 -1, 0 -1))")
    assert(collect(1).getAs[GeometryUDT](0).toString == "POLYGON ((19.292893218813454 20.707106781186546, 19.4444297669804 20.831469612302545, 19.61731656763491 20.923879532511286, 19.804909677983872 20.980785280403232, 20 21, 20.195090322016128 20.980785280403232, 20.38268343236509 20.923879532511286, 20.5555702330196 20.831469612302545, 20.707106781186546 20.707106781186546, 20.831469612302545 20.5555702330196, 20.923879532511286 20.38268343236509, 20.980785280403232 20.195090322016128, 21 20, 20.980785280403232 19.804909677983872, 20.923879532511286 19.61731656763491, 20.831469612302545 19.4444297669804, 20.707106781186546 19.292893218813454, 0.7071067811865475 -0.7071067811865475, 0.5555702330196023 -0.8314696123025452, 0.3826834323650898 -0.9238795325112867, 0.1950903220161283 -0.9807852804032304, 0.0000000000000001 -1, -0.1950903220161282 -0.9807852804032304, -0.3826834323650897 -0.9238795325112867, -0.555570233019602 -0.8314696123025453, -0.7071067811865475 -0.7071067811865476, -0.8314696123025453 -0.5555702330196022, -0.9238795325112867 -0.3826834323650899, -0.9807852804032304 -0.1950903220161286, -1 -0.0000000000000001, -0.9807852804032304 0.1950903220161284, -0.9238795325112866 0.3826834323650901, -0.8314696123025449 0.5555702330196026, -0.7071067811865475 0.7071067811865475, 19.292893218813454 20.707106781186546))")
    assert(collect(2).getAs[GeometryUDT](0).toString == "POLYGON ((1 0, 0.9807852804032304 -0.1950903220161282, 0.9238795325112867 -0.3826834323650898, 0.8314696123025452 -0.5555702330196022, 0.7071067811865476 -0.7071067811865475, 0.5555702330196023 -0.8314696123025452, 0.3826834323650898 -0.9238795325112867, 0.1950903220161283 -0.9807852804032304, 0.0000000000000001 -1, -0.1950903220161282 -0.9807852804032304, -0.3826834323650897 -0.9238795325112867, -0.555570233019602 -0.8314696123025453, -0.7071067811865475 -0.7071067811865476, -0.8314696123025453 -0.5555702330196022, -0.9238795325112868 -0.3826834323650894, -0.9807852804032305 -0.1950903220161277, -1 0.0000000000000008, -0.9807852804032302 0.1950903220161292, -0.9238795325112863 0.3826834323650909, -0.8314696123025445 0.5555702330196034, -0.7071067811865464 0.7071067811865487, -0.5555702330196007 0.8314696123025462, -0.3826834323650879 0.9238795325112875, -0.1950903220161261 0.9807852804032309, 0.0000000000000025 1, 0.1950903220161309 0.9807852804032299, 0.3826834323650924 0.9238795325112856, 0.5555702330196048 0.8314696123025435, 0.7071067811865499 0.7071067811865451, 0.8314696123025472 0.5555702330195993, 0.9238795325112882 0.3826834323650863, 0.9807852804032312 0.1950903220161244, 1 0))")
    assert(collect(3).getAs[GeometryUDT](0).toString == "POLYGON EMPTY")
    assert(collect(4).getAs[GeometryUDT](0).toString == "POLYGON ((-0.7071067811865475 0.7071067811865475, 0.2928932188134525 1.7071067811865475, 0.444429766980398 1.8314696123025453, 0.6173165676349103 1.9238795325112867, 0.8049096779838718 1.9807852804032304, 1 2, 1.1950903220161284 1.9807852804032304, 1.3826834323650898 1.9238795325112867, 1.5555702330196022 1.8314696123025453, 1.7071067811865475 1.7071067811865475, 1.8314696123025453 1.5555702330196022, 1.9238795325112867 1.3826834323650898, 1.9807852804032304 1.1950903220161282, 2 1, 2 0, 1.9807852804032304 -0.1950903220161282, 1.9238795325112867 -0.3826834323650898, 1.8314696123025453 -0.5555702330196022, 1.7071067811865475 -0.7071067811865475, 1.5555702330196022 -0.8314696123025452, 1.3826834323650898 -0.9238795325112867, 1.1950903220161284 -0.9807852804032304, 1 -1, 0 -1, -0.1950903220161284 -0.9807852804032304, -0.3826834323650897 -0.9238795325112867, -0.555570233019602 -0.8314696123025453, -0.7071067811865475 -0.7071067811865476, -0.8314696123025453 -0.5555702330196022, -0.9238795325112867 -0.3826834323650899, -0.9807852804032304 -0.1950903220161286, -1 -0.0000000000000001, -0.9807852804032304 0.1950903220161284, -0.9238795325112868 0.3826834323650897, -0.8314696123025455 0.555570233019602, -0.7071067811865475 0.7071067811865475))")

    val rst2 = df.select(st_buffer(col("geo"), lit(1)))
    rst2.show(false)

    val collect2 = rst2.collect()

    assert(collect2(0).getAs[GeometryUDT](0).toString == "POLYGON ((0 -1, -0.1950903220161287 -0.9807852804032303, -0.3826834323650903 -0.9238795325112865, -0.5555702330196022 -0.8314696123025452, -0.7071067811865477 -0.7071067811865475, -0.8314696123025455 -0.555570233019602, -0.9238795325112868 -0.3826834323650897, -0.9807852804032304 -0.1950903220161284, -1 0, -1 1, -0.9807852804032304 1.1950903220161286, -0.9238795325112867 1.3826834323650898, -0.8314696123025453 1.5555702330196022, -0.7071067811865475 1.7071067811865475, -0.555570233019602 1.8314696123025453, -0.3826834323650897 1.9238795325112867, -0.1950903220161282 1.9807852804032304, 0 2, 1 2, 1.1950903220161284 1.9807852804032304, 1.3826834323650898 1.9238795325112867, 1.5555702330196022 1.8314696123025453, 1.7071067811865475 1.7071067811865475, 1.8314696123025453 1.5555702330196022, 1.9238795325112867 1.3826834323650898, 1.9807852804032304 1.1950903220161282, 2 1, 2 0, 1.9807852804032304 -0.1950903220161282, 1.9238795325112867 -0.3826834323650898, 1.8314696123025453 -0.5555702330196022, 1.7071067811865475 -0.7071067811865475, 1.5555702330196022 -0.8314696123025452, 1.3826834323650898 -0.9238795325112867, 1.1950903220161284 -0.9807852804032304, 1 -1, 0 -1))")
    assert(collect2(1).getAs[GeometryUDT](0).toString == "POLYGON ((19.292893218813454 20.707106781186546, 19.4444297669804 20.831469612302545, 19.61731656763491 20.923879532511286, 19.804909677983872 20.980785280403232, 20 21, 20.195090322016128 20.980785280403232, 20.38268343236509 20.923879532511286, 20.5555702330196 20.831469612302545, 20.707106781186546 20.707106781186546, 20.831469612302545 20.5555702330196, 20.923879532511286 20.38268343236509, 20.980785280403232 20.195090322016128, 21 20, 20.980785280403232 19.804909677983872, 20.923879532511286 19.61731656763491, 20.831469612302545 19.4444297669804, 20.707106781186546 19.292893218813454, 0.7071067811865475 -0.7071067811865475, 0.5555702330196023 -0.8314696123025452, 0.3826834323650898 -0.9238795325112867, 0.1950903220161283 -0.9807852804032304, 0.0000000000000001 -1, -0.1950903220161282 -0.9807852804032304, -0.3826834323650897 -0.9238795325112867, -0.555570233019602 -0.8314696123025453, -0.7071067811865475 -0.7071067811865476, -0.8314696123025453 -0.5555702330196022, -0.9238795325112867 -0.3826834323650899, -0.9807852804032304 -0.1950903220161286, -1 -0.0000000000000001, -0.9807852804032304 0.1950903220161284, -0.9238795325112866 0.3826834323650901, -0.8314696123025449 0.5555702330196026, -0.7071067811865475 0.7071067811865475, 19.292893218813454 20.707106781186546))")
    assert(collect2(2).getAs[GeometryUDT](0).toString == "POLYGON ((1 0, 0.9807852804032304 -0.1950903220161282, 0.9238795325112867 -0.3826834323650898, 0.8314696123025452 -0.5555702330196022, 0.7071067811865476 -0.7071067811865475, 0.5555702330196023 -0.8314696123025452, 0.3826834323650898 -0.9238795325112867, 0.1950903220161283 -0.9807852804032304, 0.0000000000000001 -1, -0.1950903220161282 -0.9807852804032304, -0.3826834323650897 -0.9238795325112867, -0.555570233019602 -0.8314696123025453, -0.7071067811865475 -0.7071067811865476, -0.8314696123025453 -0.5555702330196022, -0.9238795325112868 -0.3826834323650894, -0.9807852804032305 -0.1950903220161277, -1 0.0000000000000008, -0.9807852804032302 0.1950903220161292, -0.9238795325112863 0.3826834323650909, -0.8314696123025445 0.5555702330196034, -0.7071067811865464 0.7071067811865487, -0.5555702330196007 0.8314696123025462, -0.3826834323650879 0.9238795325112875, -0.1950903220161261 0.9807852804032309, 0.0000000000000025 1, 0.1950903220161309 0.9807852804032299, 0.3826834323650924 0.9238795325112856, 0.5555702330196048 0.8314696123025435, 0.7071067811865499 0.7071067811865451, 0.8314696123025472 0.5555702330195993, 0.9238795325112882 0.3826834323650863, 0.9807852804032312 0.1950903220161244, 1 0))")
    assert(collect2(3).getAs[GeometryUDT](0).toString == "POLYGON EMPTY")
    assert(collect2(4).getAs[GeometryUDT](0).toString == "POLYGON ((-0.7071067811865475 0.7071067811865475, 0.2928932188134525 1.7071067811865475, 0.444429766980398 1.8314696123025453, 0.6173165676349103 1.9238795325112867, 0.8049096779838718 1.9807852804032304, 1 2, 1.1950903220161284 1.9807852804032304, 1.3826834323650898 1.9238795325112867, 1.5555702330196022 1.8314696123025453, 1.7071067811865475 1.7071067811865475, 1.8314696123025453 1.5555702330196022, 1.9238795325112867 1.3826834323650898, 1.9807852804032304 1.1950903220161282, 2 1, 2 0, 1.9807852804032304 -0.1950903220161282, 1.9238795325112867 -0.3826834323650898, 1.8314696123025453 -0.5555702330196022, 1.7071067811865475 -0.7071067811865475, 1.5555702330196022 -0.8314696123025452, 1.3826834323650898 -0.9238795325112867, 1.1950903220161284 -0.9807852804032304, 1 -1, 0 -1, -0.1950903220161284 -0.9807852804032304, -0.3826834323650897 -0.9238795325112867, -0.555570233019602 -0.8314696123025453, -0.7071067811865475 -0.7071067811865476, -0.8314696123025453 -0.5555702330196022, -0.9238795325112867 -0.3826834323650899, -0.9807852804032304 -0.1950903220161286, -1 -0.0000000000000001, -0.9807852804032304 0.1950903220161284, -0.9238795325112868 0.3826834323650897, -0.8314696123025455 0.555570233019602, -0.7071067811865475 0.7071067811865475))")
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
    rst.show(false)

    //    rst.queryExecution.debug.codegen()

    val collect = rst.collect()

    assert(collect(0).isNullAt(0))
    assert(collect(1).getAs[GeometryUDT](0).toString == "POLYGON ((19.292893218813454 20.707106781186546, 19.4444297669804 20.831469612302545, 19.61731656763491 20.923879532511286, 19.804909677983872 20.980785280403232, 20 21, 20.195090322016128 20.980785280403232, 20.38268343236509 20.923879532511286, 20.5555702330196 20.831469612302545, 20.707106781186546 20.707106781186546, 20.831469612302545 20.5555702330196, 20.923879532511286 20.38268343236509, 20.980785280403232 20.195090322016128, 21 20, 20.980785280403232 19.804909677983872, 20.923879532511286 19.61731656763491, 20.831469612302545 19.4444297669804, 20.707106781186546 19.292893218813454, 0.7071067811865475 -0.7071067811865475, 0.5555702330196023 -0.8314696123025452, 0.3826834323650898 -0.9238795325112867, 0.1950903220161283 -0.9807852804032304, 0.0000000000000001 -1, -0.1950903220161282 -0.9807852804032304, -0.3826834323650897 -0.9238795325112867, -0.555570233019602 -0.8314696123025453, -0.7071067811865475 -0.7071067811865476, -0.8314696123025453 -0.5555702330196022, -0.9238795325112867 -0.3826834323650899, -0.9807852804032304 -0.1950903220161286, -1 -0.0000000000000001, -0.9807852804032304 0.1950903220161284, -0.9238795325112866 0.3826834323650901, -0.8314696123025449 0.5555702330196026, -0.7071067811865475 0.7071067811865475, 19.292893218813454 20.707106781186546))")
    assert(collect(2).isNullAt(0))
    assert(collect(3).getAs[GeometryUDT](0).toString == "POLYGON ((0.2928932188134525 2.7071067811865475, 4.292893218813452 6.707106781186548, 4.168530387697455 6.555570233019603, 4.076120467488714 6.3826834323650905, 4.01921471959677 6.195090322016129, 4 6, 4.01921471959677 5.804909677983872, 4.076120467488713 5.61731656763491, 4.168530387697454 5.444429766980398, 4.292893218813452 5.292893218813452, 4.444429766980398 5.168530387697455, 4.6173165676349095 5.076120467488714, 4.804909677983871 5.01921471959677, 5 5, 5.195090322016128 5.01921471959677, 5.38268343236509 5.076120467488713, 5.555570233019602 5.168530387697454, 5.707106781186548 5.292893218813452, 1.7071067811865475 1.2928932188134525, 1.831469612302545 1.4444297669803974, 1.9238795325112865 1.61731656763491, 1.9807852804032304 1.8049096779838716, 2 2, 1.9807852804032304 2.1950903220161284, 1.9238795325112867 2.3826834323650896, 1.8314696123025453 2.555570233019602, 1.7071067811865475 2.7071067811865475, 1.5555702330196022 2.8314696123025453, 1.3826834323650898 2.923879532511287, 1.1950903220161284 2.9807852804032304, 1 3, 0.8049096779838718 2.9807852804032304, 0.6173165676349103 2.923879532511287, 0.444429766980398 2.8314696123025453, 0.2928932188134525 2.7071067811865475))")
    assert(collect(4).getAs[GeometryUDT](0).toString == "POLYGON ((-0.7071067811865475 0.7071067811865475, 0.2928932188134525 1.7071067811865475, 0.444429766980398 1.8314696123025453, 0.6173165676349103 1.9238795325112867, 0.8049096779838718 1.9807852804032304, 1 2, 1.1950903220161284 1.9807852804032304, 1.3826834323650898 1.9238795325112867, 1.5555702330196022 1.8314696123025453, 1.7071067811865475 1.7071067811865475, 1.8314696123025453 1.5555702330196022, 1.9238795325112867 1.3826834323650898, 1.9807852804032304 1.1950903220161282, 2 1, 2 0, 1.9807852804032304 -0.1950903220161282, 1.9238795325112867 -0.3826834323650898, 1.8314696123025453 -0.5555702330196022, 1.7071067811865475 -0.7071067811865475, 1.5555702330196022 -0.8314696123025452, 1.3826834323650898 -0.9238795325112867, 1.1950903220161284 -0.9807852804032304, 1 -1, 0 -1, -0.1950903220161284 -0.9807852804032304, -0.3826834323650897 -0.9238795325112867, -0.555570233019602 -0.8314696123025453, -0.7071067811865475 -0.7071067811865476, -0.8314696123025453 -0.5555702330196022, -0.9238795325112867 -0.3826834323650899, -0.9807852804032304 -0.1950903220161286, -1 -0.0000000000000001, -0.9807852804032304 0.1950903220161284, -0.9238795325112868 0.3826834323650897, -0.8314696123025455 0.555570233019602, -0.7071067811865475 0.7071067811865475))")

    val rst2 = df.select(st_buffer(st_geomfromtext(col("geo")), lit(1)))
    rst2.show(false)

    val collect2 = rst2.collect()

    assert(collect2(0).isNullAt(0))
    assert(collect2(1).getAs[GeometryUDT](0).toString == "POLYGON ((19.292893218813454 20.707106781186546, 19.4444297669804 20.831469612302545, 19.61731656763491 20.923879532511286, 19.804909677983872 20.980785280403232, 20 21, 20.195090322016128 20.980785280403232, 20.38268343236509 20.923879532511286, 20.5555702330196 20.831469612302545, 20.707106781186546 20.707106781186546, 20.831469612302545 20.5555702330196, 20.923879532511286 20.38268343236509, 20.980785280403232 20.195090322016128, 21 20, 20.980785280403232 19.804909677983872, 20.923879532511286 19.61731656763491, 20.831469612302545 19.4444297669804, 20.707106781186546 19.292893218813454, 0.7071067811865475 -0.7071067811865475, 0.5555702330196023 -0.8314696123025452, 0.3826834323650898 -0.9238795325112867, 0.1950903220161283 -0.9807852804032304, 0.0000000000000001 -1, -0.1950903220161282 -0.9807852804032304, -0.3826834323650897 -0.9238795325112867, -0.555570233019602 -0.8314696123025453, -0.7071067811865475 -0.7071067811865476, -0.8314696123025453 -0.5555702330196022, -0.9238795325112867 -0.3826834323650899, -0.9807852804032304 -0.1950903220161286, -1 -0.0000000000000001, -0.9807852804032304 0.1950903220161284, -0.9238795325112866 0.3826834323650901, -0.8314696123025449 0.5555702330196026, -0.7071067811865475 0.7071067811865475, 19.292893218813454 20.707106781186546))")
    assert(collect2(2).isNullAt(0))
    assert(collect2(3).getAs[GeometryUDT](0).toString == "POLYGON ((0.2928932188134525 2.7071067811865475, 4.292893218813452 6.707106781186548, 4.168530387697455 6.555570233019603, 4.076120467488714 6.3826834323650905, 4.01921471959677 6.195090322016129, 4 6, 4.01921471959677 5.804909677983872, 4.076120467488713 5.61731656763491, 4.168530387697454 5.444429766980398, 4.292893218813452 5.292893218813452, 4.444429766980398 5.168530387697455, 4.6173165676349095 5.076120467488714, 4.804909677983871 5.01921471959677, 5 5, 5.195090322016128 5.01921471959677, 5.38268343236509 5.076120467488713, 5.555570233019602 5.168530387697454, 5.707106781186548 5.292893218813452, 1.7071067811865475 1.2928932188134525, 1.831469612302545 1.4444297669803974, 1.9238795325112865 1.61731656763491, 1.9807852804032304 1.8049096779838716, 2 2, 1.9807852804032304 2.1950903220161284, 1.9238795325112867 2.3826834323650896, 1.8314696123025453 2.555570233019602, 1.7071067811865475 2.7071067811865475, 1.5555702330196022 2.8314696123025453, 1.3826834323650898 2.923879532511287, 1.1950903220161284 2.9807852804032304, 1 3, 0.8049096779838718 2.9807852804032304, 0.6173165676349103 2.923879532511287, 0.444429766980398 2.8314696123025453, 0.2928932188134525 2.7071067811865475))")
    assert(collect2(4).getAs[GeometryUDT](0).toString == "POLYGON ((-0.7071067811865475 0.7071067811865475, 0.2928932188134525 1.7071067811865475, 0.444429766980398 1.8314696123025453, 0.6173165676349103 1.9238795325112867, 0.8049096779838718 1.9807852804032304, 1 2, 1.1950903220161284 1.9807852804032304, 1.3826834323650898 1.9238795325112867, 1.5555702330196022 1.8314696123025453, 1.7071067811865475 1.7071067811865475, 1.8314696123025453 1.5555702330196022, 1.9238795325112867 1.3826834323650898, 1.9807852804032304 1.1950903220161282, 2 1, 2 0, 1.9807852804032304 -0.1950903220161282, 1.9238795325112867 -0.3826834323650898, 1.8314696123025453 -0.5555702330196022, 1.7071067811865475 -0.7071067811865475, 1.5555702330196022 -0.8314696123025452, 1.3826834323650898 -0.9238795325112867, 1.1950903220161284 -0.9807852804032304, 1 -1, 0 -1, -0.1950903220161284 -0.9807852804032304, -0.3826834323650897 -0.9238795325112867, -0.555570233019602 -0.8314696123025453, -0.7071067811865475 -0.7071067811865476, -0.8314696123025453 -0.5555702330196022, -0.9238795325112867 -0.3826834323650899, -0.9807852804032304 -0.1950903220161286, -1 -0.0000000000000001, -0.9807852804032304 0.1950903220161284, -0.9238795325112868 0.3826834323650897, -0.8314696123025455 0.555570233019602, -0.7071067811865475 0.7071067811865475))")
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

    val rst = spark.sql("select ST_PrecisionReduce(geo, 1 + 1) from data ")
    rst.show(false)

    //    rst.queryExecution.debug.codegen()

    val collect = rst.collect()

    assert(collect(0).getAs[GeometryUDT](0).toString == "POLYGON ((0 0, 0 1.5, 1.5 1.5, 1.5 0, 0 0))")
    assert(collect(1).getAs[GeometryUDT](0).toString == "LINESTRING (0 0, 10 10.5, 20 20.5)")
    assert(collect(2).getAs[GeometryUDT](0).toString == "POINT (0 0.5)")
    assert(collect(3).getAs[GeometryUDT](0).toString == "POLYGON EMPTY")
    assert(collect(4).getAs[GeometryUDT](0).toString == "MULTIPOLYGON (((0 0, 1 0, 1 1, 0 0)))")

    val rst2 = df.select(st_precisionreduce(col("geo"), lit(2)))
    rst2.show(false)

    val collect2 = rst2.collect()

    assert(collect2(0).getAs[GeometryUDT](0).toString == "POLYGON ((0 0, 0 1.5, 1.5 1.5, 1.5 0, 0 0))")
    assert(collect2(1).getAs[GeometryUDT](0).toString == "LINESTRING (0 0, 10 10.5, 20 20.5)")
    assert(collect2(2).getAs[GeometryUDT](0).toString == "POINT (0 0.5)")
    assert(collect2(3).getAs[GeometryUDT](0).toString == "POLYGON EMPTY")
    assert(collect2(4).getAs[GeometryUDT](0).toString == "MULTIPOLYGON (((0 0, 1 0, 1 1, 0 0)))")
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
    rst.show(false)

    //    rst.queryExecution.debug.codegen()

    val collect = rst.collect()

    assert(collect(0).isNullAt(0))
    assert(collect(1).getAs[GeometryUDT](0).toString == "LINESTRING (0 0, 10 10.5, 20 20.5)")
    assert(collect(2).isNullAt(0))
    assert(collect(3).getAs[GeometryUDT](0).toString == "POLYGON EMPTY")
    assert(collect(4).getAs[GeometryUDT](0).toString == "MULTIPOLYGON (((0 0, 1 0, 1 1, 0 0)))")

    val rst2 = df.select(st_precisionreduce(st_geomfromtext(col("geo")), lit(2)))
    rst2.show(false)

    val collect2 = rst2.collect()

    assert(collect2(0).isNullAt(0))
    assert(collect2(1).getAs[GeometryUDT](0).toString == "LINESTRING (0 0, 10 10.5, 20 20.5)")
    assert(collect2(2).isNullAt(0))
    assert(collect2(3).getAs[GeometryUDT](0).toString == "POLYGON EMPTY")
    assert(collect2(4).getAs[GeometryUDT](0).toString == "MULTIPOLYGON (((0 0, 1 0, 1 1, 0 0)))")
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
    rst.show(false)

    //    rst.queryExecution.debug.codegen()

    val collect = rst.collect()

    assert(collect(0).getAs[GeometryUDT](0).toString == "POINT (20 20)")
    assert(collect(1).getAs[GeometryUDT](0).toString == "POLYGON ((10 10, 10 20, 20 20, 20 10, 10 10))")
    assert(collect(2).getAs[GeometryUDT](0).toString == "POLYGON ((40 10, 10 10, 10 40, 40 40, 40 10))")

    val rst2 = df.select(st_intersection(col("left_geo"), col("right_geo")))
    rst2.show(false)

    val collect2 = rst2.collect()

    assert(collect2(0).getAs[GeometryUDT](0).toString == "POINT (20 20)")
    assert(collect2(1).getAs[GeometryUDT](0).toString == "POLYGON ((10 10, 10 20, 20 20, 20 10, 10 10))")
    assert(collect2(2).getAs[GeometryUDT](0).toString == "POLYGON ((40 10, 10 10, 10 40, 40 40, 40 10))")
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
    rst.show(false)

    //    rst.queryExecution.debug.codegen()

    val collect = rst.collect()

    assert(collect(0).isNullAt(0))
    assert(collect(1).isNullAt(0))
    assert(collect(2).isNullAt(0))
    assert(collect(3).getAs[GeometryUDT](0).toString == "POLYGON ((40 10, 10 10, 10 40, 40 40, 40 10))")

    val rst2 = df.select(st_intersection(st_geomfromtext(col("left_geo")), st_geomfromtext(col("right_geo"))))
    rst2.show(false)

    val collect2 = rst2.collect()

    assert(collect2(0).isNullAt(0))
    assert(collect2(1).isNullAt(0))
    assert(collect2(2).isNullAt(0))
    assert(collect2(3).getAs[GeometryUDT](0).toString == "POLYGON ((40 10, 10 10, 10 40, 40 40, 40 10))")
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

    val rst = spark.sql("select ST_SimplifyPreserveTopology(geo, 1 + 1) from data ")
    rst.show(false)

    //    rst.queryExecution.debug.codegen()

    val collect = rst.collect()

    assert(collect(0).getAs[GeometryUDT](0).toString == "POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))")
    assert(collect(1).getAs[GeometryUDT](0).toString == "LINESTRING (0 0, 20 20)")
    assert(collect(2).getAs[GeometryUDT](0).toString == "POINT (0 0)")
    assert(collect(3).getAs[GeometryUDT](0).toString == "POLYGON EMPTY")
    assert(collect(4).getAs[GeometryUDT](0).toString == "POLYGON ((0 0, 1 0, 1 1, 0 0))")

    val rst2 = df.select(st_simplifypreservetopology(col("geo"), lit(2)))
    rst2.show(false)

    val collect2 = rst2.collect()

    assert(collect2(0).getAs[GeometryUDT](0).toString == "POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))")
    assert(collect2(1).getAs[GeometryUDT](0).toString == "LINESTRING (0 0, 20 20)")
    assert(collect2(2).getAs[GeometryUDT](0).toString == "POINT (0 0)")
    assert(collect2(3).getAs[GeometryUDT](0).toString == "POLYGON EMPTY")
    assert(collect2(4).getAs[GeometryUDT](0).toString == "POLYGON ((0 0, 1 0, 1 1, 0 0))")
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
    rst.show(false)

    //    rst.queryExecution.debug.codegen()

    val collect = rst.collect()

    assert(collect(0).isNullAt(0))
    assert(collect(1).getAs[GeometryUDT](0).toString == "LINESTRING (0 0, 20 20)")
    assert(collect(2).isNullAt(0))
    assert(collect(3).getAs[GeometryUDT](0).toString == "POLYGON ((1 2, 3 4, 5 6, 1 2))")
    assert(collect(4).getAs[GeometryUDT](0).toString == "POLYGON ((0 0, 1 0, 1 1, 0 0))")

    val rst2 = df.select(st_simplifypreservetopology(st_geomfromtext(col("geo")), lit(2)))
    rst2.show(false)

    val collect2 = rst2.collect()

    assert(collect2(0).isNullAt(0))
    assert(collect2(1).getAs[GeometryUDT](0).toString == "LINESTRING (0 0, 20 20)")
    assert(collect2(2).isNullAt(0))
    assert(collect2(3).getAs[GeometryUDT](0).toString == "POLYGON ((1 2, 3 4, 5 6, 1 2))")
    assert(collect2(4).getAs[GeometryUDT](0).toString == "POLYGON ((0 0, 1 0, 1 1, 0 0))")
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
    rst.show(false)

    //    rst.queryExecution.debug.codegen()

    val collect = rst.collect()

    assert(collect(0).getAs[GeometryUDT](0).toString == "POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))")
    assert(collect(1).getAs[GeometryUDT](0).toString == "LINESTRING (0 0, 20 20)")
    assert(collect(2).getAs[GeometryUDT](0).toString == "POINT (0 0)")
    assert(collect(3).getAs[GeometryUDT](0).toString == "GEOMETRYCOLLECTION EMPTY")
    assert(collect(4).getAs[GeometryUDT](0).toString == "POLYGON ((0 0, 1 1, 1 0, 0 0))")

    val rst2 = df.select(st_convexhull(col("geo")))
    rst2.show(false)

    val collect2 = rst2.collect()

    assert(collect2(0).getAs[GeometryUDT](0).toString == "POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))")
    assert(collect2(1).getAs[GeometryUDT](0).toString == "LINESTRING (0 0, 20 20)")
    assert(collect2(2).getAs[GeometryUDT](0).toString == "POINT (0 0)")
    assert(collect2(3).getAs[GeometryUDT](0).toString == "GEOMETRYCOLLECTION EMPTY")
    assert(collect2(4).getAs[GeometryUDT](0).toString == "POLYGON ((0 0, 1 1, 1 0, 0 0))")
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
    rst.show(false)

    //    rst.queryExecution.debug.codegen()

    val collect = rst.collect()

    assert(collect(0).isNullAt(0))
    assert(collect(1).getAs[GeometryUDT](0).toString == "LINESTRING (0 0, 20 20)")
    assert(collect(2).isNullAt(0))
    assert(collect(3).getAs[GeometryUDT](0).toString == "LINESTRING (1 2, 5 6)")
    assert(collect(4).getAs[GeometryUDT](0).toString == "POLYGON ((0 0, 1 1, 1 0, 0 0))")

    val rst2 = df.select(st_convexhull(st_geomfromtext(col("geo"))))
    rst2.show(false)

    val collect2 = rst2.collect()

    assert(collect2(0).isNullAt(0))
    assert(collect2(1).getAs[GeometryUDT](0).toString == "LINESTRING (0 0, 20 20)")
    assert(collect2(2).isNullAt(0))
    assert(collect2(3).getAs[GeometryUDT](0).toString == "LINESTRING (1 2, 5 6)")
    assert(collect2(4).getAs[GeometryUDT](0).toString == "POLYGON ((0 0, 1 1, 1 0, 0 0))")
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
    rst.show(false)

    //    rst.queryExecution.debug.codegen()

    val collect = rst.collect()

    assert(collect(0).getDouble(0) == 1.0)
    assert(collect(1).getDouble(0) == 0.0)
    assert(collect(2).getDouble(0) == 0.0)
    assert(collect(3).getDouble(0) == 0.0)
    assert(collect(4).getDouble(0) == 0.5)

    val rst2 = df.select(st_area(col("geo")))
    rst2.show(false)

    val collect2 = rst2.collect()

    assert(collect2(0).getDouble(0) == 1.0)
    assert(collect2(1).getDouble(0) == 0.0)
    assert(collect2(2).getDouble(0) == 0.0)
    assert(collect2(3).getDouble(0) == 0.0)
    assert(collect2(4).getDouble(0) == 0.5)
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
    rst.show(false)

    //    rst.queryExecution.debug.codegen()

    val collect = rst.collect()

    assert(collect(0).isNullAt(0))
    assert(collect(1).getDouble(0) == 0.0)
    assert(collect(2).isNullAt(0))
    assert(collect(3).getDouble(0) == 0.0)
    assert(collect(4).getDouble(0) == 0.5)

    val rst2 = df.select(st_area(st_geomfromtext(col("geo"))))
    rst2.show(false)

    val collect2 = rst2.collect()

    assert(collect2(0).isNullAt(0))
    assert(collect2(1).getDouble(0) == 0.0)
    assert(collect2(2).isNullAt(0))
    assert(collect2(3).getDouble(0) == 0.0)
    assert(collect2(4).getDouble(0) == 0.5)
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
    rst.show(false)

    //    rst.queryExecution.debug.codegen()

    val collect = rst.collect()

    assert(collect(0).getDouble(0) == 4.0)
    assert(collect(1).getDouble(0) == 28.284271247461902)
    assert(collect(2).getDouble(0) == 0.0)
    assert(collect(3).getDouble(0) == 0.0)
    assert(collect(4).getDouble(0) == 3.414213562373095)

    val rst2 = df.select(st_length(col("geo")))
    rst2.show(false)

    val collect2 = rst2.collect()

    assert(collect2(0).getDouble(0) == 4.0)
    assert(collect2(1).getDouble(0) == 28.284271247461902)
    assert(collect2(2).getDouble(0) == 0.0)
    assert(collect2(3).getDouble(0) == 0.0)
    assert(collect2(4).getDouble(0) == 3.414213562373095)
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
    rst.show(false)

    //    rst.queryExecution.debug.codegen()

    val collect = rst.collect()

    assert(collect(0).isNullAt(0))
    assert(collect(1).getDouble(0) == 28.284271247461902)
    assert(collect(2).isNullAt(0))
    assert(collect(3).getDouble(0) == 11.313708498984761)
    assert(collect(4).getDouble(0) == 3.414213562373095)

    val rst2 = df.select(st_length(st_geomfromtext(col("geo"))))
    rst2.show(false)

    val collect2 = rst2.collect()

    assert(collect2(0).isNullAt(0))
    assert(collect2(1).getDouble(0) == 28.284271247461902)
    assert(collect2(2).isNullAt(0))
    assert(collect2(3).getDouble(0) == 11.313708498984761)
    assert(collect2(4).getDouble(0) == 3.414213562373095)
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
    rst.show(false)

    //    rst.queryExecution.debug.codegen()

    val collect = rst.collect()

    assert(collect(0).getDouble(0) == 28.284271247461902)
    assert(collect(1).getDouble(0) == 70.71067811865476)
    assert(collect(2).getDouble(0) == 28.284271247461902)
    assert(collect(3).getDouble(0) == 14.142135623730951)

    val rst2 = df.select(st_hausdorffdistance(col("left_geo"), col("right_geo")))
    rst2.show(false)

    val collect2 = rst2.collect()

    assert(collect2(0).getDouble(0) == 28.284271247461902)
    assert(collect2(1).getDouble(0) == 70.71067811865476)
    assert(collect2(2).getDouble(0) == 28.284271247461902)
    assert(collect2(3).getDouble(0) == 14.142135623730951)
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
    rst.show(false)

    //    rst.queryExecution.debug.codegen()

    val collect = rst.collect()

    assert(collect(0).isNullAt(0))
    assert(collect(1).isNullAt(0))
    assert(collect(2).isNullAt(0))
    assert(collect(3).getDouble(0) == 14.142135623730951)

    val rst2 = df.select(st_hausdorffdistance(st_geomfromtext(col("left_geo")), st_geomfromtext(col("right_geo"))))

    rst2.show(false)

    val collect2 = rst2.collect()

    assert(collect2(0).isNullAt(0))
    assert(collect2(1).isNullAt(0))
    assert(collect2(2).isNullAt(0))
    assert(collect2(3).getDouble(0) == 14.142135623730951)
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
    rst.show(false)

    //    rst.queryExecution.debug.codegen()

    val collect = rst.collect()

    assert(collect(0).getDouble(0) == 0.0)
    assert(collect(1).getDouble(0) == 14.142135623730951)
    assert(collect(2).getDouble(0) == 0.0)
    assert(collect(3).getDouble(0) == 0.0)

    val rst2 = df.select(st_distance(col("left_geo"), col("right_geo")))
    rst2.show(false)

    val collect2 = rst2.collect()

    assert(collect2(0).getDouble(0) == 0.0)
    assert(collect2(1).getDouble(0) == 14.142135623730951)
    assert(collect2(2).getDouble(0) == 0.0)
    assert(collect2(3).getDouble(0) == 0.0)
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
    rst.show(false)

    //    rst.queryExecution.debug.codegen()

    val collect = rst.collect()

    assert(collect(0).isNullAt(0))
    assert(collect(1).isNullAt(0))
    assert(collect(2).isNullAt(0))
    assert(collect(3).getDouble(0) == 0.0)

    val rst2 = df.select(st_distance(st_geomfromtext(col("left_geo")), st_geomfromtext(col("right_geo"))))
    rst2.show(false)

    val collect2 = rst2.collect()

    assert(collect2(0).isNullAt(0))
    assert(collect2(1).isNullAt(0))
    assert(collect2(2).isNullAt(0))
    assert(collect2(3).getDouble(0) == 0.0)
  }

  test("ST_Equals") {
    val data = Seq(
      Row(GeometryUDT.FromWkt("POINT (20 20)"), GeometryUDT.FromWkt("POINT (20 20)")),
      Row(GeometryUDT.FromWkt("POINT (50 50)"), GeometryUDT.FromWkt("POLYGON ((0 0, 40 0, 40 40, 0 40, 0 0))")),
      Row(GeometryUDT.FromWkt("POLYGON ((10 10, 20 10, 20 20, 10 20, 10 10))"), GeometryUDT.FromWkt("POLYGON ((0 0, 40 0, 40 40, 0 40, 0 0))")),
      Row(GeometryUDT.FromWkt("POLYGON ((10 10, 50 10, 50 50, 10 50, 10 10))"), GeometryUDT.FromWkt("POLYGON ((10 10, 50 10, 50 50, 10 50, 10 10))")),
      Row(GeometryUDT.FromWkt("LINESTRING (0 0, 5 5, 10 10)"), GeometryUDT.FromWkt("LINESTRING (0 0, 10 10)")),
      Row(GeometryUDT.FromWkt("LINESTRING (10 10, 0 0)"), GeometryUDT.FromWkt("LINESTRING (0 0, 10 10)")),
      Row(GeometryUDT.FromWkt("LINESTRING (0 0, 1 1)"), GeometryUDT.FromWkt("LINESTRING (1 1, 0 0)"))
    )

    val schema = StructType(Array(StructField("left_geo", new GeometryUDT, nullable = true), StructField("right_geo", new GeometryUDT, nullable = true)))
    val df = spark.createDataFrame(spark.sparkContext.parallelize(data), schema)
    df.createOrReplaceTempView("data")

    val rst = spark.sql("select ST_Equals(left_geo, right_geo) from data ")
    rst.show(false)

    //    rst.queryExecution.debug.codegen()

    val collect = rst.collect()

    assert(collect(0).getBoolean(0) == true)
    assert(collect(1).getBoolean(0) == false)
    assert(collect(2).getBoolean(0) == false)
    assert(collect(3).getBoolean(0) == true)
    assert(collect(4).getBoolean(0) == true)
    assert(collect(5).getBoolean(0) == true)
    assert(collect(6).getBoolean(0) == true)

    val rst2 = df.select(st_equals(col("left_geo"), col("right_geo")))
    rst2.show(false)

    val collect2 = rst2.collect()

    assert(collect2(0).getBoolean(0) == true)
    assert(collect2(1).getBoolean(0) == false)
    assert(collect2(2).getBoolean(0) == false)
    assert(collect2(3).getBoolean(0) == true)
    assert(collect2(4).getBoolean(0) == true)
    assert(collect2(5).getBoolean(0) == true)
    assert(collect2(6).getBoolean(0) == true)
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
    rst.show(false)

    //    rst.queryExecution.debug.codegen()

    val collect = rst.collect()

    assert(collect(0).isNullAt(0))
    assert(collect(1).isNullAt(0))
    assert(collect(2).isNullAt(0))
    assert(collect(3).getBoolean(0) == true)

    val rst2 = df.select(st_equals(st_geomfromtext(col("left_geo")), st_geomfromtext(col("right_geo"))))
    rst2.show(false)

    val collect2 = rst2.collect()

    assert(collect2(0).isNullAt(0))
    assert(collect2(1).isNullAt(0))
    assert(collect2(2).isNullAt(0))
    assert(collect2(3).getBoolean(0) == true)
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
    rst.show(false)

    //    rst.queryExecution.debug.codegen()

    val collect = rst.collect()

    assert(collect(0).getBoolean(0) == true)
    assert(collect(1).getBoolean(0) == false)
    assert(collect(2).getBoolean(0) == true)
    assert(collect(3).getBoolean(0) == false)

    val rst2 = df.select(st_touches(col("left_geo"), col("right_geo")))
    rst2.show(false)

    val collect2 = rst2.collect()

    assert(collect2(0).getBoolean(0) == true)
    assert(collect2(1).getBoolean(0) == false)
    assert(collect2(2).getBoolean(0) == true)
    assert(collect2(3).getBoolean(0) == false)
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
    rst.show(false)

    //    rst.queryExecution.debug.codegen()

    val collect = rst.collect()

    assert(collect(0).isNullAt(0))
    assert(collect(1).isNullAt(0))
    assert(collect(2).isNullAt(0))
    assert(collect(3).getBoolean(0) == false)

    val rst2 = df.select(st_touches(st_geomfromtext(col("left_geo")), st_geomfromtext(col("right_geo"))))
    rst2.show(false)

    val collect2 = rst2.collect()

    assert(collect2(0).isNullAt(0))
    assert(collect2(1).isNullAt(0))
    assert(collect2(2).isNullAt(0))
    assert(collect2(3).getBoolean(0) == false)
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
    rst.show(false)

    //    rst.queryExecution.debug.codegen()

    val collect = rst.collect()

    assert(collect(0).getBoolean(0) == false)
    assert(collect(1).getBoolean(0) == false)
    assert(collect(2).getBoolean(0) == false)
    assert(collect(3).getBoolean(0) == true)

    val rst2 = df.select(st_overlaps(col("left_geo"), col("right_geo")))
    rst2.show(false)

    val collect2 = rst2.collect()

    assert(collect2(0).getBoolean(0) == false)
    assert(collect2(1).getBoolean(0) == false)
    assert(collect2(2).getBoolean(0) == false)
    assert(collect2(3).getBoolean(0) == true)
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
    rst.show(false)

    //    rst.queryExecution.debug.codegen()

    val collect = rst.collect()

    assert(collect(0).isNullAt(0))
    assert(collect(1).isNullAt(0))
    assert(collect(2).isNullAt(0))
    assert(collect(3).getBoolean(0) == true)

    val rst2 = df.select(st_overlaps(st_geomfromtext(col("left_geo")), st_geomfromtext(col("right_geo"))))
    rst2.show(false)

    val collect2 = rst2.collect()

    assert(collect2(0).isNullAt(0))
    assert(collect2(1).isNullAt(0))
    assert(collect2(2).isNullAt(0))
    assert(collect2(3).getBoolean(0) == true)
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
    rst.show(false)

    //    rst.queryExecution.debug.codegen()

    val collect = rst.collect()

    assert(collect(0).getBoolean(0) == true)
    assert(collect(1).getBoolean(0) == false)
    assert(collect(2).getBoolean(0) == false)
    assert(collect(3).getBoolean(0) == false)

    val rst2 = df.select(st_crosses(col("left_geo"), col("right_geo")))
    rst2.show(false)

    val collect2 = rst2.collect()

    assert(collect2(0).getBoolean(0) == true)
    assert(collect2(1).getBoolean(0) == false)
    assert(collect2(2).getBoolean(0) == false)
    assert(collect2(3).getBoolean(0) == false)
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
    rst.show(false)

    //    rst.queryExecution.debug.codegen()

    val collect = rst.collect()

    assert(collect(0).isNullAt(0))
    assert(collect(1).isNullAt(0))
    assert(collect(2).isNullAt(0))
    assert(collect(3).getBoolean(0) == false)

    val rst2 = df.select(st_crosses(st_geomfromtext(col("left_geo")), st_geomfromtext(col("right_geo"))))
    rst2.show(false)

    val collect2 = rst2.collect()

    assert(collect2(0).isNullAt(0))
    assert(collect2(1).isNullAt(0))
    assert(collect2(2).isNullAt(0))
    assert(collect2(3).getBoolean(0) == false)
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
    rst.show(false)

    //    rst.queryExecution.debug.codegen()

    val collect = rst.collect()

    assert(collect(0).getBoolean(0) == false)
    assert(collect(1).getBoolean(0) == true)
    assert(collect(2).getBoolean(0) == false)
    assert(collect(3).getBoolean(0) == true)

    val rst2 = df.select(st_contains(col("left_geo"), col("right_geo")))
    rst2.show(false)

    val collect2 = rst2.collect()

    assert(collect2(0).getBoolean(0) == false)
    assert(collect2(1).getBoolean(0) == true)
    assert(collect2(2).getBoolean(0) == false)
    assert(collect2(3).getBoolean(0) == true)
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
    rst.show(false)

    //    rst.queryExecution.debug.codegen()

    val collect = rst.collect()

    assert(collect(0).isNullAt(0))
    assert(collect(1).isNullAt(0))
    assert(collect(2).isNullAt(0))
    assert(collect(3).getBoolean(0) == false)

    val rst2 = df.select(st_contains(st_geomfromtext(col("left_geo")), st_geomfromtext(col("right_geo"))))
    rst2.show(false)

    val collect2 = rst2.collect()

    assert(collect2(0).isNullAt(0))
    assert(collect2(1).isNullAt(0))
    assert(collect2(2).isNullAt(0))
    assert(collect2(3).getBoolean(0) == false)
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
    rst.show(false)

    //    rst.queryExecution.debug.codegen()

    val collect = rst.collect()

    assert(collect(0).getBoolean(0) == true)
    assert(collect(1).getBoolean(0) == false)
    assert(collect(2).getBoolean(0) == true)
    assert(collect(3).getBoolean(0) == true)

    val rst2 = df.select(st_intersects(col("left_geo"), col("right_geo")))
    rst2.show(false)

    val collect2 = rst2.collect()

    assert(collect2(0).getBoolean(0) == true)
    assert(collect2(1).getBoolean(0) == false)
    assert(collect2(2).getBoolean(0) == true)
    assert(collect2(3).getBoolean(0) == true)
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
    rst.show(false)

    //    rst.queryExecution.debug.codegen()

    val collect = rst.collect()

    assert(collect(0).isNullAt(0))
    assert(collect(1).isNullAt(0))
    assert(collect(2).isNullAt(0))
    assert(collect(3).getBoolean(0) == true)

    val rst2 = df.select(st_intersects(st_geomfromtext(col("left_geo")), st_geomfromtext(col("right_geo"))))
    rst2.show(false)

    val collect2 = rst2.collect()

    assert(collect2(0).isNullAt(0))
    assert(collect2(1).isNullAt(0))
    assert(collect2(2).isNullAt(0))
    assert(collect2(3).getBoolean(0) == true)
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
    rst.show(false)

    //    rst.queryExecution.debug.codegen()

    val collect = rst.collect()

    assert(collect(0).getDouble(0) == 1531.6176273715332)
    assert(collect(1).getDouble(0) == 137894.9747266781)

    val rst2 = df.select(st_distancesphere(col("left_geo"), col("right_geo")))
    rst2.show(false)

    val collect2 = rst2.collect()

    assert(collect2(0).getDouble(0) == 1531.6176273715332)
    assert(collect2(1).getDouble(0) == 137894.9747266781)
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
    rst.show(false)

    //    rst.queryExecution.debug.codegen()

    val collect = rst.collect()

    assert(collect(0).isNullAt(0))
    assert(collect(1).isNullAt(0))
    assert(collect(2).isNullAt(0))
    assert(collect(3).getDouble(0) == -1.0)
    assert(collect(4).getDouble(0) == -1.0)

    val rst2 = df.select(st_distancesphere(st_geomfromtext(col("left_geo")), st_geomfromtext(col("right_geo"))))
    rst2.show(false)

    val collect2 = rst2.collect()

    assert(collect2(0).isNullAt(0))
    assert(collect2(1).isNullAt(0))
    assert(collect2(2).isNullAt(0))
    assert(collect2(3).getDouble(0) == -1.0)
    assert(collect2(4).getDouble(0) == -1.0)
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
    rst.show(false)

    //    rst.queryExecution.debug.codegen()

    val collect = rst.collect()

    assert(collect(0).getAs[GeometryUDT](0).toString == "POINT (-8235544.280259263 4974337.520006327)")
    assert(collect(1).getAs[GeometryUDT](0).toString == "POINT (-8251391.611649103 4947867.503249774)")

    val rst2 = df.select(st_transform(col("geo"), lit("EPSG:4326"), lit("EPSG:3857")))
    rst2.show(false)

    val collect2 = rst2.collect()

    assert(collect2(0).getAs[GeometryUDT](0).toString == "POINT (-8235544.280259263 4974337.520006327)")
    assert(collect2(1).getAs[GeometryUDT](0).toString == "POINT (-8251391.611649103 4947867.503249774)")
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
    rst.show(false)

    //    rst.queryExecution.debug.codegen()

    val collect = rst.collect()

    assert(collect(0).isNullAt(0))
    assert(collect(1).getAs[GeometryUDT](0).toString == "POINT (-8236547.714149274 4972580.886197065)")
    assert(collect(2).getAs[GeometryUDT](0).toString == "POLYGON ((0 -0.0000000007081155, 4452779.631730943 -0.0000000007081155, 4452779.631730943 4865942.279503176, 0 4865942.279503176, 0 -0.0000000007081155))")
    assert(collect(3).getAs[GeometryUDT](0).toString == "POINT (-8235544.280259263 4974337.520006327)")

    val rst2 = df.select(st_transform(st_geomfromtext(col("geo")), lit("EPSG:4326"), lit("EPSG:3857")))
    rst2.show(false)

    val collect2 = rst2.collect()

    assert(collect2(0).isNullAt(0))
    assert(collect2(1).getAs[GeometryUDT](0).toString == "POINT (-8236547.714149274 4972580.886197065)")
    assert(collect2(2).getAs[GeometryUDT](0).toString == "POLYGON ((0 -0.0000000007081155, 4452779.631730943 -0.0000000007081155, 4452779.631730943 4865942.279503176, 0 4865942.279503176, 0 -0.0000000007081155))")
    assert(collect2(3).getAs[GeometryUDT](0).toString == "POINT (-8235544.280259263 4974337.520006327)")
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
    rst.show(false)

    //    rst.queryExecution.debug.codegen()

    val collect = rst.collect()

    assert(collect(0).getAs[GeometryUDT](0).toString == "POLYGON ((0 0, 0 1, 1 2, 0 0))")
    assert(collect(1).getAs[GeometryUDT](0).toString == "MULTIPOLYGON (((0 0, 0 4, 4 4, 4 0, 0 0)), ((0 0, 4 0, 4 1, 0 1, 0 0)))")

    val rst2 = df.select(st_makevalid(col("geo")))
    rst2.show(false)

    val collect2 = rst2.collect()

    assert(collect2(0).getAs[GeometryUDT](0).toString == "POLYGON ((0 0, 0 1, 1 2, 0 0))")
    assert(collect2(1).getAs[GeometryUDT](0).toString == "MULTIPOLYGON (((0 0, 0 4, 4 4, 4 0, 0 0)), ((0 0, 4 0, 4 1, 0 1, 0 0)))")
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
    rst.show(false)

    //    rst.queryExecution.debug.codegen()

    val collect = rst.collect()

    assert(collect(0).isNullAt(0))
    assert(collect(1).getAs[GeometryUDT](0).toString == "POINT (-73.990167 40.729884)")
    assert(collect(2).getAs[GeometryUDT](0).toString == "POLYGON ((0 0, 40 0, 40 40, 0 40, 0 0))")
    assert(collect(3).getAs[GeometryUDT](0).toString == "MULTIPOLYGON (((0 0, 0 4, 4 4, 4 0, 0 0)), ((0 0, 4 0, 4 1, 0 1, 0 0)))")

    val rst2 = df.select(st_makevalid(st_geomfromtext(col("geo"))))
    rst2.show(false)

    val collect2 = rst2.collect()

    assert(collect2(0).isNullAt(0))
    assert(collect2(1).getAs[GeometryUDT](0).toString == "POINT (-73.990167 40.729884)")
    assert(collect2(2).getAs[GeometryUDT](0).toString == "POLYGON ((0 0, 40 0, 40 40, 0 40, 0 0))")
    assert(collect2(3).getAs[GeometryUDT](0).toString == "MULTIPOLYGON (((0 0, 0 4, 4 4, 4 0, 0 0)), ((0 0, 4 0, 4 1, 0 1, 0 0)))")
  }

  test("ST_CurveToLine") {
    val data = Seq(
      Row(GeometryUDT.FromWkt("CURVEPOLYGON(CIRCULARSTRING(0 0, 4 0, 4 4, 0 4, 0 0))")),
    )

    val schema = StructType(Array(StructField("geo", new GeometryUDT, nullable = true)))
    val df = spark.createDataFrame(spark.sparkContext.parallelize(data), schema)
    df.createOrReplaceTempView("data")

    val rst = spark.sql("select ST_CurveToLine(geo) from data ")
    rst.show(false)

    //    rst.queryExecution.debug.codegen()

    val collect = rst.collect()

    assert(collect(0).getAs[GeometryUDT](0).toString.substring(0, 7) == "POLYGON")

    val rst2 = df.select(st_makevalid(col("geo")))
    rst2.show(false)

    val collect2 = rst2.collect()

    assert(collect2(0).getAs[GeometryUDT](0).toString.substring(0, 7) == "POLYGON")
  }

  test("ST_CurveToLine-Null") {
    val data = Seq(
      Row(null),
      Row("CURVEPOLYGON(CIRCULARSTRING(0 0, 4 0, 4 4, 0 4, 0 0))"),
    )

    val schema = StructType(Array(StructField("geo", StringType, nullable = true)))
    val df = spark.createDataFrame(spark.sparkContext.parallelize(data), schema)
    df.createOrReplaceTempView("data")

    val rst = spark.sql("select ST_CurveToLine(ST_GeomFromText(geo)) from data ")
    rst.show(false)

    //    rst.queryExecution.debug.codegen()

    val collect = rst.collect()

    assert(collect(0).isNullAt(0))
    assert(collect(1).getAs[GeometryUDT](0).toString.substring(0, 7) == "POLYGON")

    val rst2 = df.select(st_makevalid(st_geomfromtext(col("geo"))))
    rst2.show(false)

    val collect2 = rst2.collect()

    assert(collect2(0).isNullAt(0))
    assert(collect2(1).getAs[GeometryUDT](0).toString.substring(0, 7) == "POLYGON")
  }

  test("ST_Translate") {
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

    val rst = spark.sql("select ST_Translate(geo, 1 + 1, 2) from data ")
    rst.show(false)

    //    rst.queryExecution.debug.codegen()

    val collect = rst.collect()

    assert(collect(0).getAs[GeometryUDT](0).toString == "POLYGON ((2 2, 2 3, 3 3, 3 2, 2 2))")
    assert(collect(1).getAs[GeometryUDT](0).toString == "LINESTRING (2 2, 12 12, 22 22)")
    assert(collect(2).getAs[GeometryUDT](0).toString == "POINT (2 2)")
    assert(collect(3).getAs[GeometryUDT](0).toString == "POLYGON EMPTY")
    assert(collect(4).getAs[GeometryUDT](0).toString == "MULTIPOLYGON (((2 2, 3 2, 3 3, 2 2)))")

    val rst2 = df.select(st_translate(col("geo"), lit(2), lit(2.0)))
    rst2.show(false)

    val collect2 = rst2.collect()

    assert(collect2(0).getAs[GeometryUDT](0).toString == "POLYGON ((2 2, 2 3, 3 3, 3 2, 2 2))")
    assert(collect2(1).getAs[GeometryUDT](0).toString == "LINESTRING (2 2, 12 12, 22 22)")
    assert(collect2(2).getAs[GeometryUDT](0).toString == "POINT (2 2)")
    assert(collect2(3).getAs[GeometryUDT](0).toString == "POLYGON EMPTY")
    assert(collect2(4).getAs[GeometryUDT](0).toString == "MULTIPOLYGON (((2 2, 3 2, 3 3, 2 2)))")
  }

  test("ST_Translate-Null") {
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

    val rst = spark.sql("select ST_Translate(ST_GeomFromText(geo), 1 + 1, 2) from data ")
    rst.show(false)

    //    rst.queryExecution.debug.codegen()

    val collect = rst.collect()

    assert(collect(0).isNullAt(0))
    assert(collect(1).getAs[GeometryUDT](0).toString == "LINESTRING (2 2, 12 12, 22 22)")
    assert(collect(2).isNullAt(0))
    assert(collect(3).getAs[GeometryUDT](0).toString == "POLYGON ((3 4, 5 6, 7 8, 3 4))")
    assert(collect(4).getAs[GeometryUDT](0).toString == "MULTIPOLYGON (((2 2, 3 2, 3 3, 2 2)))")

    val rst2 = df.select(st_translate(st_geomfromtext(col("geo")), lit(2), lit(2)))
    rst2.show(false)

    val collect2 = rst2.collect()

    assert(collect2(0).isNullAt(0))
    assert(collect2(1).getAs[GeometryUDT](0).toString == "LINESTRING (2 2, 12 12, 22 22)")
    assert(collect2(2).isNullAt(0))
    assert(collect2(3).getAs[GeometryUDT](0).toString == "POLYGON ((3 4, 5 6, 7 8, 3 4))")
    assert(collect2(4).getAs[GeometryUDT](0).toString == "MULTIPOLYGON (((2 2, 3 2, 3 3, 2 2)))")
  }

  test("ST_Rotate") {
    val data = Seq(
      Row(GeometryUDT.FromWkt("POINT (1 6)")),
      Row(GeometryUDT.FromWkt("LINESTRING (0 0, 0 1, 1 1)")),
      Row(GeometryUDT.FromWkt("LINESTRING (0 0, 1 0, 1 1, 0 0)")),
      Row(GeometryUDT.FromWkt("POLYGON ((0 0, 0 1, 1 1, 0 0))")),
      Row(GeometryUDT.FromWkt("MULTIPOINT (0 0, 1 0, 1 2, 1 2)")),
      Row(GeometryUDT.FromWkt("MULTILINESTRING ( (0 0, 1 2), (0 0, 1 0, 1 1),(-1 2,3 4,1 -3,-2 1) )")),
      Row(GeometryUDT.FromWkt("MULTIPOLYGON ( ((0 0, 1 4, 1 0,0 0)) )")),
    )

    val schema = StructType(Array(StructField("geo", new GeometryUDT, nullable = true)))
    val df = spark.createDataFrame(spark.sparkContext.parallelize(data), schema)
    df.createOrReplaceTempView("data")

    val rst = spark.sql("select ST_PrecisionReduce(ST_Rotate(geo, CAST(2 * acos(0.0) AS FLOAT), 1, 0), 2) from data ")
    rst.show(false)

    //    rst.queryExecution.debug.codegen()

    val collect = rst.collect()

    assert(collect(0).getAs[GeometryUDT](0).toString == "POINT (1 -6)")
    assert(collect(1).getAs[GeometryUDT](0).toString == "LINESTRING (2 0, 2 -1, 1 -1)")
    assert(collect(2).getAs[GeometryUDT](0).toString == "LINESTRING (2 0, 1 0, 1 -1, 2 0)")
    assert(collect(3).getAs[GeometryUDT](0).toString == "POLYGON ((2 0, 2 -1, 1 -1, 2 0))")
    assert(collect(4).getAs[GeometryUDT](0).toString == "MULTIPOINT ((2 0), (1 0), (1 -2), (1 -2))")
    assert(collect(5).getAs[GeometryUDT](0).toString == "MULTILINESTRING ((2 0, 1 -2), (2 0, 1 0, 1 -1), (3 -2, -1 -4, 1 3, 4 -1))")
    assert(collect(6).getAs[GeometryUDT](0).toString == "MULTIPOLYGON (((2 0, 1 -4, 1 0, 2 0)))")

    val rst2 = df.select(st_precisionreduce(st_rotate(col("geo"), lit(2 * scala.math.acos(0.0)), lit(1), lit(0)), lit(2)))
    rst2.show(false)

    val collect2 = rst2.collect()

    assert(collect2(0).getAs[GeometryUDT](0).toString == "POINT (1 -6)")
    assert(collect2(1).getAs[GeometryUDT](0).toString == "LINESTRING (2 0, 2 -1, 1 -1)")
    assert(collect2(2).getAs[GeometryUDT](0).toString == "LINESTRING (2 0, 1 0, 1 -1, 2 0)")
    assert(collect2(3).getAs[GeometryUDT](0).toString == "POLYGON ((2 0, 2 -1, 1 -1, 2 0))")
    assert(collect2(4).getAs[GeometryUDT](0).toString == "MULTIPOINT ((2 0), (1 0), (1 -2), (1 -2))")
    assert(collect2(5).getAs[GeometryUDT](0).toString == "MULTILINESTRING ((2 0, 1 -2), (2 0, 1 0, 1 -1), (3 -2, -1 -4, 1 3, 4 -1))")
    assert(collect2(6).getAs[GeometryUDT](0).toString == "MULTIPOLYGON (((2 0, 1 -4, 1 0, 2 0)))")
  }

  test("ST_Rotate-Null") {
    val data = Seq(
      Row("POINT (1 6)"),
      Row("LINESTRING (0 0, 0 1, 1 1)"),
      Row("LINESTRING (0 0, 1 0, 1 1, 0 0)"),
      Row("POLYGON ((0 0, 0 1, 1 1, 0 0))"),
      Row("MULTIPOINT (0 0, 1 0, 1 2, 1 2)"),
      Row("MULTILINESTRING ( (0 0, 1 2), (0 0, 1 0, 1 1),(-1 2,3 4,1 -3,-2 1) )"),
      Row("MULTIPOLYGON ( ((0 0, 1 4, 1 0,0 0)) )"),
      Row(null),
      Row("error geometry"),
    )

    val schema = StructType(Array(StructField("geo", StringType, nullable = true)))
    val df = spark.createDataFrame(spark.sparkContext.parallelize(data), schema)
    df.createOrReplaceTempView("data")

    val rst = spark.sql("select ST_PrecisionReduce(ST_Rotate(ST_GeomFromText(geo), CAST(2 * acos(0.0) AS FLOAT), 1, 0), 2) from data ")
    rst.show(false)

    //    rst.queryExecution.debug.codegen()

    val collect = rst.collect()

    assert(collect(0).getAs[GeometryUDT](0).toString == "POINT (1 -6)")
    assert(collect(1).getAs[GeometryUDT](0).toString == "LINESTRING (2 0, 2 -1, 1 -1)")
    assert(collect(2).getAs[GeometryUDT](0).toString == "LINESTRING (2 0, 1 0, 1 -1, 2 0)")
    assert(collect(3).getAs[GeometryUDT](0).toString == "POLYGON ((2 0, 2 -1, 1 -1, 2 0))")
    assert(collect(4).getAs[GeometryUDT](0).toString == "MULTIPOINT ((2 0), (1 0), (1 -2), (1 -2))")
    assert(collect(5).getAs[GeometryUDT](0).toString == "MULTILINESTRING ((2 0, 1 -2), (2 0, 1 0, 1 -1), (3 -2, -1 -4, 1 3, 4 -1))")
    assert(collect(6).getAs[GeometryUDT](0).toString == "MULTIPOLYGON (((2 0, 1 -4, 1 0, 2 0)))")
    assert(collect(7).isNullAt(0))
    assert(collect(8).isNullAt(0))

    val rst2 = df.select(st_precisionreduce(st_rotate(st_geomfromtext(col("geo")), lit(2 * scala.math.acos(0.0)), lit(1), lit(0)), lit(2)))
    rst2.show(false)

    val collect2 = rst2.collect()

    assert(collect2(0).getAs[GeometryUDT](0).toString == "POINT (1 -6)")
    assert(collect2(1).getAs[GeometryUDT](0).toString == "LINESTRING (2 0, 2 -1, 1 -1)")
    assert(collect2(2).getAs[GeometryUDT](0).toString == "LINESTRING (2 0, 1 0, 1 -1, 2 0)")
    assert(collect2(3).getAs[GeometryUDT](0).toString == "POLYGON ((2 0, 2 -1, 1 -1, 2 0))")
    assert(collect2(4).getAs[GeometryUDT](0).toString == "MULTIPOINT ((2 0), (1 0), (1 -2), (1 -2))")
    assert(collect2(5).getAs[GeometryUDT](0).toString == "MULTILINESTRING ((2 0, 1 -2), (2 0, 1 0, 1 -1), (3 -2, -1 -4, 1 3, 4 -1))")
    assert(collect2(6).getAs[GeometryUDT](0).toString == "MULTIPOLYGON (((2 0, 1 -4, 1 0, 2 0)))")
    assert(collect2(7).isNullAt(0))
    assert(collect2(8).isNullAt(0))
  }

  test("ST_SymDifference") {
    val data = Seq(
      Row(GeometryUDT.FromWkt("LINESTRING (0 0,5 0)"), GeometryUDT.FromWkt("LINESTRING (4 0,6 0)")),
    )

    val schema = StructType(Array(StructField("left_geo", new GeometryUDT, nullable = true), StructField("right_geo", new GeometryUDT, nullable = true)))
    val df = spark.createDataFrame(spark.sparkContext.parallelize(data), schema)
    df.createOrReplaceTempView("data")

    val rst = spark.sql("select ST_SymDifference(left_geo, right_geo) from data ")
    rst.show(false)

    //    rst.queryExecution.debug.codegen()

    val collect = rst.collect()

    assert(collect(0).getAs[GeometryUDT](0).toString == "MULTILINESTRING ((0 0, 4 0), (5 0, 6 0))")

    val rst2 = df.select(st_symdifference(col("left_geo"), col("right_geo")))
    rst2.show(false)

    val collect2 = rst2.collect()

    assert(collect2(0).getAs[GeometryUDT](0).toString == "MULTILINESTRING ((0 0, 4 0), (5 0, 6 0))")
  }

  test("ST_SymDifference-Null") {
    val data = Seq(
      Row("LINESTRING (0 0,5 0)", "LINESTRING (4 0,6 0)"),
      Row(null, "LINESTRING (4 0,6 0)"),
      Row(null, null),
    )

    val schema = StructType(Array(StructField("left_geo", StringType, nullable = true), StructField("right_geo", StringType, nullable = true)))
    val df = spark.createDataFrame(spark.sparkContext.parallelize(data), schema)
    df.createOrReplaceTempView("data")

    val rst = spark.sql("select ST_SymDifference(ST_GeomFromText(left_geo), ST_GeomFromText(right_geo)) from data ")
    rst.show(false)

    //    rst.queryExecution.debug.codegen()

    val collect = rst.collect()

    assert(collect(0).getAs[GeometryUDT](0).toString == "MULTILINESTRING ((0 0, 4 0), (5 0, 6 0))")
    assert(collect(1).isNullAt(0))
    assert(collect(2).isNullAt(0))

    val rst2 = df.select(st_symdifference(st_geomfromtext(col("left_geo")), st_geomfromtext(col("right_geo"))))
    rst2.show(false)

    val collect2 = rst2.collect()

    assert(collect2(0).getAs[GeometryUDT](0).toString == "MULTILINESTRING ((0 0, 4 0), (5 0, 6 0))")
    assert(collect2(1).isNullAt(0))
    assert(collect2(2).isNullAt(0))
  }

  test("ST_Difference") {
    val data = Seq(
      Row(GeometryUDT.FromWkt("LINESTRING (0 0,5 0)"), GeometryUDT.FromWkt("LINESTRING (4 0,6 0)")),
    )

    val schema = StructType(Array(StructField("left_geo", new GeometryUDT, nullable = true), StructField("right_geo", new GeometryUDT, nullable = true)))
    val df = spark.createDataFrame(spark.sparkContext.parallelize(data), schema)
    df.createOrReplaceTempView("data")

    val rst = spark.sql("select ST_Difference(left_geo, right_geo) from data ")
    rst.show(false)

    //    rst.queryExecution.debug.codegen()

    val collect = rst.collect()

    assert(collect(0).getAs[GeometryUDT](0).toString == "LINESTRING (0 0, 4 0)")

    val rst2 = df.select(st_difference(col("left_geo"), col("right_geo")))
    rst2.show(false)

    val collect2 = rst2.collect()

    assert(collect2(0).getAs[GeometryUDT](0).toString == "LINESTRING (0 0, 4 0)")
  }

  test("ST_Difference-Null") {
    val data = Seq(
      Row("LINESTRING (0 0,5 0)", "LINESTRING (4 0,6 0)"),
      Row(null, "LINESTRING (4 0,6 0)"),
      Row(null, null),
    )

    val schema = StructType(Array(StructField("left_geo", StringType, nullable = true), StructField("right_geo", StringType, nullable = true)))
    val df = spark.createDataFrame(spark.sparkContext.parallelize(data), schema)
    df.createOrReplaceTempView("data")

    val rst = spark.sql("select ST_Difference(ST_GeomFromText(left_geo), ST_GeomFromText(right_geo)) from data ")
    rst.show(false)

    //    rst.queryExecution.debug.codegen()

    val collect = rst.collect()

    assert(collect(0).getAs[GeometryUDT](0).toString == "LINESTRING (0 0, 4 0)")
    assert(collect(1).isNullAt(0))
    assert(collect(2).isNullAt(0))

    val rst2 = df.select(st_difference(st_geomfromtext(col("left_geo")), st_geomfromtext(col("right_geo"))))
    rst2.show(false)

    val collect2 = rst2.collect()

    assert(collect2(0).getAs[GeometryUDT](0).toString == "LINESTRING (0 0, 4 0)")
    assert(collect2(1).isNullAt(0))
    assert(collect2(2).isNullAt(0))
  }

  test("ST_Union") {
    val data = Seq(
      Row(GeometryUDT.FromWkt("POINT (0 1)"), GeometryUDT.FromWkt("POLYGON ((0 0,0 2,2 2,0 0))")),
      Row(GeometryUDT.FromWkt("LINESTRING (0 0, 0 1, 1 1)"), GeometryUDT.FromWkt("LINESTRING (0 0, 0 1, 1 2)")),
      Row(GeometryUDT.FromWkt("LINESTRING (0 0, 1 0, 1 1, 0 0)"), GeometryUDT.FromWkt("POINT (2 3)")),
      Row(GeometryUDT.FromWkt("POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))"), GeometryUDT.FromWkt("MULTIPOINT (0 0, 1 0, 1 2, 1 2)")),
      Row(GeometryUDT.FromWkt("MULTIPOINT (0 0, 1 0, 1 2, 1 2)"), GeometryUDT.FromWkt("MULTILINESTRING ( (0 0, 1 2), (0 0, 1 0, 1 1),(-1 2,3 4,1 -3,-2 1) )")),
      Row(GeometryUDT.FromWkt("MULTILINESTRING ( (0 0, 1 2), (0 0, 1 0, 1 1),(-1 2,3 4,1 -3,-2 1) )"), GeometryUDT.FromWkt("MULTIPOLYGON ( ((0 0, 1 4, 1 0,0 0)) )")),
      Row(GeometryUDT.FromWkt("MULTIPOLYGON ( ((0 0, 1 4, 1 0,0 0)) )"), GeometryUDT.FromWkt("POINT (1 5)")),
    )

    val schema = StructType(Array(StructField("left_geo", new GeometryUDT, nullable = true), StructField("right_geo", new GeometryUDT, nullable = true)))
    val df = spark.createDataFrame(spark.sparkContext.parallelize(data), schema)
    df.createOrReplaceTempView("data")

    val rst = spark.sql("select ST_Union(left_geo, right_geo) from data ")
    rst.show(false)

    //    rst.queryExecution.debug.codegen()

    val collect = rst.collect()

    assert(collect(0).getAs[GeometryUDT](0).toString == "POLYGON ((0 0, 0 2, 2 2, 0 0))")
    assert(collect(1).getAs[GeometryUDT](0).toString == "MULTILINESTRING ((0 0, 0 1), (0 1, 1 1), (0 1, 1 2))")
    assert(collect(2).getAs[GeometryUDT](0).toString == "GEOMETRYCOLLECTION (POINT (2 3), LINESTRING (0 0, 1 0, 1 1, 0 0))")
    assert(collect(3).getAs[GeometryUDT](0).toString == "GEOMETRYCOLLECTION (POINT (1 2), POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0)))")
    assert(collect(4).getAs[GeometryUDT](0).toString == "MULTILINESTRING ((0 0, 1 2), (0 0, 1 0, 1 1), (-1 2, 3 4, 1 -3, -2 1))")
    assert(collect(5).getAs[GeometryUDT](0).toString == "GEOMETRYCOLLECTION (LINESTRING (-1 2, 0.7142857142857143 2.857142857142857), LINESTRING (1 3, 3 4, 1 -3, -2 1), POLYGON ((1 0, 0 0, 0.7142857142857143 2.857142857142857, 1 4, 1 3, 1 2, 1 1, 1 0)))")
    assert(collect(6).getAs[GeometryUDT](0).toString == "GEOMETRYCOLLECTION (POINT (1 5), POLYGON ((0 0, 1 4, 1 0, 0 0)))")

    val rst2 = df.select(st_union(col("left_geo"), col("right_geo")))
    rst2.show(false)

    val collect2 = rst2.collect()

    assert(collect2(0).getAs[GeometryUDT](0).toString == "POLYGON ((0 0, 0 2, 2 2, 0 0))")
    assert(collect2(1).getAs[GeometryUDT](0).toString == "MULTILINESTRING ((0 0, 0 1), (0 1, 1 1), (0 1, 1 2))")
    assert(collect2(2).getAs[GeometryUDT](0).toString == "GEOMETRYCOLLECTION (POINT (2 3), LINESTRING (0 0, 1 0, 1 1, 0 0))")
    assert(collect2(3).getAs[GeometryUDT](0).toString == "GEOMETRYCOLLECTION (POINT (1 2), POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0)))")
    assert(collect2(4).getAs[GeometryUDT](0).toString == "MULTILINESTRING ((0 0, 1 2), (0 0, 1 0, 1 1), (-1 2, 3 4, 1 -3, -2 1))")
    assert(collect2(5).getAs[GeometryUDT](0).toString == "GEOMETRYCOLLECTION (LINESTRING (-1 2, 0.7142857142857143 2.857142857142857), LINESTRING (1 3, 3 4, 1 -3, -2 1), POLYGON ((1 0, 0 0, 0.7142857142857143 2.857142857142857, 1 4, 1 3, 1 2, 1 1, 1 0)))")
    assert(collect2(6).getAs[GeometryUDT](0).toString == "GEOMETRYCOLLECTION (POINT (1 5), POLYGON ((0 0, 1 4, 1 0, 0 0)))")
  }

  test("ST_Union-Null") {
    val data = Seq(
      Row("POINT (0 1)", "POLYGON ((0 0,0 2,2 2,0 0))"),
      Row("LINESTRING (0 0, 0 1, 1 1)", "LINESTRING (0 0, 0 1, 1 2)"),
      Row("LINESTRING (0 0, 1 0, 1 1, 0 0)", "POINT (2 3)"),
      Row("POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))", "MULTIPOINT (0 0, 1 0, 1 2, 1 2)"),
      Row("MULTIPOINT (0 0, 1 0, 1 2, 1 2)", "MULTILINESTRING ( (0 0, 1 2), (0 0, 1 0, 1 1),(-1 2,3 4,1 -3,-2 1) )"),
      Row("MULTILINESTRING ( (0 0, 1 2), (0 0, 1 0, 1 1),(-1 2,3 4,1 -3,-2 1) )", "MULTIPOLYGON ( ((0 0, 1 4, 1 0,0 0)) )"),
      Row("MULTIPOLYGON ( ((0 0, 1 4, 1 0,0 0)) )", "POINT (1 5)"),
      Row(null, "POINT (1 5)"),
      Row("error geometry", "POINT (1 5)"),
    )

    val schema = StructType(Array(StructField("left_geo", StringType, nullable = true), StructField("right_geo", StringType, nullable = true)))
    val df = spark.createDataFrame(spark.sparkContext.parallelize(data), schema)
    df.createOrReplaceTempView("data")

    val rst = spark.sql("select ST_Union(ST_GeomFromText(left_geo), ST_GeomFromText(right_geo)) from data ")
    rst.show(false)

    //    rst.queryExecution.debug.codegen()

    val collect = rst.collect()

    assert(collect(0).getAs[GeometryUDT](0).toString == "POLYGON ((0 0, 0 2, 2 2, 0 0))")
    assert(collect(1).getAs[GeometryUDT](0).toString == "MULTILINESTRING ((0 0, 0 1), (0 1, 1 1), (0 1, 1 2))")
    assert(collect(2).getAs[GeometryUDT](0).toString == "GEOMETRYCOLLECTION (POINT (2 3), LINESTRING (0 0, 1 0, 1 1, 0 0))")
    assert(collect(3).getAs[GeometryUDT](0).toString == "GEOMETRYCOLLECTION (POINT (1 2), POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0)))")
    assert(collect(4).getAs[GeometryUDT](0).toString == "MULTILINESTRING ((0 0, 1 2), (0 0, 1 0, 1 1), (-1 2, 3 4, 1 -3, -2 1))")
    assert(collect(5).getAs[GeometryUDT](0).toString == "GEOMETRYCOLLECTION (LINESTRING (-1 2, 0.7142857142857143 2.857142857142857), LINESTRING (1 3, 3 4, 1 -3, -2 1), POLYGON ((1 0, 0 0, 0.7142857142857143 2.857142857142857, 1 4, 1 3, 1 2, 1 1, 1 0)))")
    assert(collect(6).getAs[GeometryUDT](0).toString == "GEOMETRYCOLLECTION (POINT (1 5), POLYGON ((0 0, 1 4, 1 0, 0 0)))")
    assert(collect(7).isNullAt(0))
    assert(collect(8).isNullAt(0))

    val rst2 = df.select(st_union(st_geomfromtext(col("left_geo")), st_geomfromtext(col("right_geo"))))
    rst2.show(false)

    val collect2 = rst2.collect()

    assert(collect2(0).getAs[GeometryUDT](0).toString == "POLYGON ((0 0, 0 2, 2 2, 0 0))")
    assert(collect2(1).getAs[GeometryUDT](0).toString == "MULTILINESTRING ((0 0, 0 1), (0 1, 1 1), (0 1, 1 2))")
    assert(collect2(2).getAs[GeometryUDT](0).toString == "GEOMETRYCOLLECTION (POINT (2 3), LINESTRING (0 0, 1 0, 1 1, 0 0))")
    assert(collect2(3).getAs[GeometryUDT](0).toString == "GEOMETRYCOLLECTION (POINT (1 2), POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0)))")
    assert(collect2(4).getAs[GeometryUDT](0).toString == "MULTILINESTRING ((0 0, 1 2), (0 0, 1 0, 1 1), (-1 2, 3 4, 1 -3, -2 1))")
    assert(collect2(5).getAs[GeometryUDT](0).toString == "GEOMETRYCOLLECTION (LINESTRING (-1 2, 0.7142857142857143 2.857142857142857), LINESTRING (1 3, 3 4, 1 -3, -2 1), POLYGON ((1 0, 0 0, 0.7142857142857143 2.857142857142857, 1 4, 1 3, 1 2, 1 1, 1 0)))")
    assert(collect2(6).getAs[GeometryUDT](0).toString == "GEOMETRYCOLLECTION (POINT (1 5), POLYGON ((0 0, 1 4, 1 0, 0 0)))")
    assert(collect2(7).isNullAt(0))
    assert(collect2(8).isNullAt(0))
  }

  test("ST_Disjoint") {
    val data = Seq(
      Row(GeometryUDT.FromWkt("POINT (0 1)"), GeometryUDT.FromWkt("POLYGON ((0 0,0 2,2 2,0 0))")),
      Row(GeometryUDT.FromWkt("LINESTRING (0 0, 0 1, 1 1)"), GeometryUDT.FromWkt("LINESTRING (0 0, 0 1, 1 2)")),
      Row(GeometryUDT.FromWkt("LINESTRING (0 0, 1 0, 1 1, 0 0)"), GeometryUDT.FromWkt("POINT (2 3)")),
      Row(GeometryUDT.FromWkt("POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))"), GeometryUDT.FromWkt("MULTIPOINT (0 0, 1 0, 1 2, 1 2)")),
      Row(GeometryUDT.FromWkt("MULTIPOINT (0 0, 1 0, 1 2, 1 2)"), GeometryUDT.FromWkt("MULTILINESTRING ( (0 0, 1 2), (0 0, 1 0, 1 1),(-1 2,3 4,1 -3,-2 1) )")),
      Row(GeometryUDT.FromWkt("MULTILINESTRING ( (0 0, 1 2), (0 0, 1 0, 1 1),(-1 2,3 4,1 -3,-2 1) )"), GeometryUDT.FromWkt("MULTIPOLYGON ( ((0 0, 1 4, 1 0,0 0)) )")),
      Row(GeometryUDT.FromWkt("MULTIPOLYGON ( ((0 0, 1 4, 1 0,0 0)) )"), GeometryUDT.FromWkt("POINT (1 5)")),
    )

    val schema = StructType(Array(StructField("left_geo", new GeometryUDT, nullable = true), StructField("right_geo", new GeometryUDT, nullable = true)))
    val df = spark.createDataFrame(spark.sparkContext.parallelize(data), schema)
    df.createOrReplaceTempView("data")

    val rst = spark.sql("select ST_Disjoint(left_geo, right_geo) from data ")
    rst.show(false)

    //    rst.queryExecution.debug.codegen()

    val collect = rst.collect()

    assert(collect(0).getBoolean(0) == false)
    assert(collect(1).getBoolean(0) == false)
    assert(collect(2).getBoolean(0) == true)
    assert(collect(3).getBoolean(0) == false)
    assert(collect(4).getBoolean(0) == false)
    assert(collect(5).getBoolean(0) == false)
    assert(collect(6).getBoolean(0) == true)

    val rst2 = df.select(st_disjoint(col("left_geo"), col("right_geo")))
    rst2.show(false)

    val collect2 = rst2.collect()

    assert(collect2(0).getBoolean(0) == false)
    assert(collect2(1).getBoolean(0) == false)
    assert(collect2(2).getBoolean(0) == true)
    assert(collect2(3).getBoolean(0) == false)
    assert(collect2(4).getBoolean(0) == false)
    assert(collect2(5).getBoolean(0) == false)
    assert(collect2(6).getBoolean(0) == true)
  }

  test("ST_Disjoint-Null") {
    val data = Seq(
      Row("POINT (0 1)", "POLYGON ((0 0,0 2,2 2,0 0))"),
      Row("LINESTRING (0 0, 0 1, 1 1)", "LINESTRING (0 0, 0 1, 1 2)"),
      Row("LINESTRING (0 0, 1 0, 1 1, 0 0)", "POINT (2 3)"),
      Row("POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))", "MULTIPOINT (0 0, 1 0, 1 2, 1 2)"),
      Row("MULTIPOINT (0 0, 1 0, 1 2, 1 2)", "MULTILINESTRING ( (0 0, 1 2), (0 0, 1 0, 1 1),(-1 2,3 4,1 -3,-2 1) )"),
      Row("MULTILINESTRING ( (0 0, 1 2), (0 0, 1 0, 1 1),(-1 2,3 4,1 -3,-2 1) )", "MULTIPOLYGON ( ((0 0, 1 4, 1 0,0 0)) )"),
      Row("MULTIPOLYGON ( ((0 0, 1 4, 1 0,0 0)) )", "POINT (1 5)"),
      Row(null, "POINT (1 5)"),
      Row("error geometry", "POINT (1 5)"),
    )

    val schema = StructType(Array(StructField("left_geo", StringType, nullable = true), StructField("right_geo", StringType, nullable = true)))
    val df = spark.createDataFrame(spark.sparkContext.parallelize(data), schema)
    df.createOrReplaceTempView("data")

    val rst = spark.sql("select ST_Disjoint(ST_GeomFromText(left_geo), ST_GeomFromText(right_geo)) from data ")
    rst.show(false)

    //    rst.queryExecution.debug.codegen()

    val collect = rst.collect()

    assert(collect(0).getBoolean(0) == false)
    assert(collect(1).getBoolean(0) == false)
    assert(collect(2).getBoolean(0) == true)
    assert(collect(3).getBoolean(0) == false)
    assert(collect(4).getBoolean(0) == false)
    assert(collect(5).getBoolean(0) == false)
    assert(collect(6).getBoolean(0) == true)
    assert(collect(7).isNullAt(0))
    assert(collect(8).isNullAt(0))

    val rst2 = df.select(st_disjoint(st_geomfromtext(col("left_geo")), st_geomfromtext(col("right_geo"))))
    rst2.show(false)

    val collect2 = rst2.collect()

    assert(collect2(0).getBoolean(0) == false)
    assert(collect2(1).getBoolean(0) == false)
    assert(collect2(2).getBoolean(0) == true)
    assert(collect2(3).getBoolean(0) == false)
    assert(collect2(4).getBoolean(0) == false)
    assert(collect2(5).getBoolean(0) == false)
    assert(collect2(6).getBoolean(0) == true)
    assert(collect2(7).isNullAt(0))
    assert(collect2(8).isNullAt(0))
  }

  test("ST_IsEmpty") {
    val data = Seq(
      Row(GeometryUDT.FromWkt("POINT (0 1)")),
      // TODO: fix Empty Points cannot be represented in WKB
      // Row(GeometryUDT.FromWkt("POINT EMPTY")),
      Row(GeometryUDT.FromWkt("LINESTRING (0 0, 0 1, 1 1)")),
      Row(GeometryUDT.FromWkt("LINESTRING EMPTY")),
      Row(GeometryUDT.FromWkt("POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))")),
      Row(GeometryUDT.FromWkt("POLYGON EMPTY")),
      Row(GeometryUDT.FromWkt("MULTIPOLYGON ( ((0 0, 1 4, 1 0,0 0)) )")),
      Row(GeometryUDT.FromWkt("MULTIPOLYGON EMPTY")),
    )

    val schema = StructType(Array(StructField("geo", new GeometryUDT, nullable = true)))
    val df = spark.createDataFrame(spark.sparkContext.parallelize(data), schema)
    df.createOrReplaceTempView("data")

    val rst = spark.sql("select ST_IsEmpty(geo) from data ")
    rst.show(false)

    //    rst.queryExecution.debug.codegen()

    val collect = rst.collect()

    assert(collect(0).getBoolean(0) == false)
    assert(collect(1).getBoolean(0) == false)
    assert(collect(2).getBoolean(0) == true)
    assert(collect(3).getBoolean(0) == false)
    assert(collect(4).getBoolean(0) == true)
    assert(collect(5).getBoolean(0) == false)
    assert(collect(6).getBoolean(0) == true)

    val rst2 = df.select(st_isempty(col("geo")))
    rst2.show(false)

    val collect2 = rst2.collect()

    assert(collect2(0).getBoolean(0) == false)
    assert(collect2(1).getBoolean(0) == false)
    assert(collect2(2).getBoolean(0) == true)
    assert(collect2(3).getBoolean(0) == false)
    assert(collect2(4).getBoolean(0) == true)
    assert(collect2(5).getBoolean(0) == false)
    assert(collect2(6).getBoolean(0) == true)
  }

  test("ST_IsEmpty-Null") {
    val data = Seq(
      Row("POINT (0 1)"),
      Row("POINT EMPTY"),
      Row("LINESTRING (0 0, 0 1, 1 1)"),
      Row("LINESTRING EMPTY"),
      Row("POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))"),
      Row("POLYGON EMPTY"),
      Row("MULTIPOLYGON ( ((0 0, 1 4, 1 0,0 0)) )"),
      Row("MULTIPOLYGON EMPTY"),
      Row(null),
      Row("error geometry"),
    )

    val schema = StructType(Array(StructField("geo", StringType, nullable = true)))
    val df = spark.createDataFrame(spark.sparkContext.parallelize(data), schema)
    df.createOrReplaceTempView("data")

    val rst = spark.sql("select ST_IsEmpty(ST_GeomFromText(geo)) from data ")
    rst.show(false)

    //    rst.queryExecution.debug.codegen()

    val collect = rst.collect()

    assert(collect(0).getBoolean(0) == false)
    assert(collect(1).getBoolean(0) == true)
    assert(collect(2).getBoolean(0) == false)
    assert(collect(3).getBoolean(0) == true)
    assert(collect(4).getBoolean(0) == false)
    assert(collect(5).getBoolean(0) == true)
    assert(collect(6).getBoolean(0) == false)
    assert(collect(7).getBoolean(0) == true)
    assert(collect(8).isNullAt(0))
    assert(collect(9).isNullAt(0))

    val rst2 = df.select(st_isempty(st_geomfromtext(col("geo"))))
    rst2.show(false)

    val collect2 = rst2.collect()

    assert(collect2(0).getBoolean(0) == false)
    assert(collect2(1).getBoolean(0) == true)
    assert(collect2(2).getBoolean(0) == false)
    assert(collect2(3).getBoolean(0) == true)
    assert(collect2(4).getBoolean(0) == false)
    assert(collect2(5).getBoolean(0) == true)
    assert(collect2(6).getBoolean(0) == false)
    assert(collect2(7).getBoolean(0) == true)
    assert(collect2(8).isNullAt(0))
    assert(collect2(9).isNullAt(0))
  }

  test("ST_Boundary") {
    val data = Seq(
      Row(GeometryUDT.FromWkt("POINT (0 1)")),
      Row(GeometryUDT.FromWkt("LINESTRING (0 0, 0 1, 1 1)")),
      Row(GeometryUDT.FromWkt("POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))")),
      Row(GeometryUDT.FromWkt("POLYGON EMPTY")),
      Row(GeometryUDT.FromWkt("MULTIPOLYGON ( ((0 0, 1 4, 1 0,0 0)) )")),
    )

    val schema = StructType(Array(StructField("geo", new GeometryUDT, nullable = true)))
    val df = spark.createDataFrame(spark.sparkContext.parallelize(data), schema)
    df.createOrReplaceTempView("data")

    val rst = spark.sql("select ST_Boundary(geo) from data ")
    rst.show(false)

    //    rst.queryExecution.debug.codegen()

    val collect = rst.collect()

    assert(collect(0).getAs[GeometryUDT](0).toString == "GEOMETRYCOLLECTION EMPTY")
    assert(collect(1).getAs[GeometryUDT](0).toString == "MULTIPOINT ((0 0), (1 1))")
    assert(collect(2).getAs[GeometryUDT](0).toString == "LINESTRING (0 0, 1 0, 1 1, 0 1, 0 0)")
    assert(collect(3).getAs[GeometryUDT](0).toString == "MULTILINESTRING EMPTY")
    assert(collect(4).getAs[GeometryUDT](0).toString == "MULTILINESTRING ((0 0, 1 4, 1 0, 0 0))")

    val rst2 = df.select(st_boundary(col("geo")))
    rst2.show(false)

    val collect2 = rst2.collect()

    assert(collect2(0).getAs[GeometryUDT](0).toString == "GEOMETRYCOLLECTION EMPTY")
    assert(collect2(1).getAs[GeometryUDT](0).toString == "MULTIPOINT ((0 0), (1 1))")
    assert(collect2(2).getAs[GeometryUDT](0).toString == "LINESTRING (0 0, 1 0, 1 1, 0 1, 0 0)")
    assert(collect2(3).getAs[GeometryUDT](0).toString == "MULTILINESTRING EMPTY")
    assert(collect2(4).getAs[GeometryUDT](0).toString == "MULTILINESTRING ((0 0, 1 4, 1 0, 0 0))")
  }

  test("ST_Boundary-Null") {
    val data = Seq(
      Row("POINT (0 1)"),
      Row("LINESTRING (0 0, 0 1, 1 1)"),
      Row("POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))"),
      Row("POLYGON EMPTY"),
      Row("MULTIPOLYGON ( ((0 0, 1 4, 1 0,0 0)) )"),
      Row(null),
      Row("error geometry"),
    )

    val schema = StructType(Array(StructField("geo", StringType, nullable = true)))
    val df = spark.createDataFrame(spark.sparkContext.parallelize(data), schema)
    df.createOrReplaceTempView("data")

    val rst = spark.sql("select ST_Boundary(ST_GeomFromText(geo)) from data ")
    rst.show(false)

    //    rst.queryExecution.debug.codegen()

    val collect = rst.collect()

    assert(collect(0).getAs[GeometryUDT](0).toString == "GEOMETRYCOLLECTION EMPTY")
    assert(collect(1).getAs[GeometryUDT](0).toString == "MULTIPOINT ((0 0), (1 1))")
    assert(collect(2).getAs[GeometryUDT](0).toString == "LINESTRING (0 0, 1 0, 1 1, 0 1, 0 0)")
    assert(collect(3).getAs[GeometryUDT](0).toString == "MULTILINESTRING EMPTY")
    assert(collect(4).getAs[GeometryUDT](0).toString == "MULTILINESTRING ((0 0, 1 4, 1 0, 0 0))")
    assert(collect(5).isNullAt(0))
    assert(collect(6).isNullAt(0))

    val rst2 = df.select(st_boundary(st_geomfromtext(col("geo"))))
    rst2.show(false)

    val collect2 = rst2.collect()

    assert(collect2(0).getAs[GeometryUDT](0).toString == "GEOMETRYCOLLECTION EMPTY")
    assert(collect2(1).getAs[GeometryUDT](0).toString == "MULTIPOINT ((0 0), (1 1))")
    assert(collect2(2).getAs[GeometryUDT](0).toString == "LINESTRING (0 0, 1 0, 1 1, 0 1, 0 0)")
    assert(collect2(3).getAs[GeometryUDT](0).toString == "MULTILINESTRING EMPTY")
    assert(collect2(4).getAs[GeometryUDT](0).toString == "MULTILINESTRING ((0 0, 1 4, 1 0, 0 0))")
    assert(collect2(5).isNullAt(0))
    assert(collect2(6).isNullAt(0))
  }

  test("ST_ExteriorRing") {
    val data = Seq(
      Row(GeometryUDT.FromWkt("POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))")),
    )

    val schema = StructType(Array(StructField("geo", new GeometryUDT, nullable = true)))
    val df = spark.createDataFrame(spark.sparkContext.parallelize(data), schema)
    df.createOrReplaceTempView("data")

    val rst = spark.sql("select ST_ExteriorRing(geo) from data ")
    rst.show(false)

    //    rst.queryExecution.debug.codegen()

    val collect = rst.collect()

    assert(collect(0).getAs[GeometryUDT](0).toString == "LINESTRING (0 0, 1 0, 1 1, 0 1, 0 0)")

    val rst2 = df.select(st_exteriorring(col("geo")))
    rst2.show(false)

    val collect2 = rst2.collect()

    assert(collect2(0).getAs[GeometryUDT](0).toString == "LINESTRING (0 0, 1 0, 1 1, 0 1, 0 0)")
  }

  test("ST_ExteriorRing-Null") {
    val data = Seq(
      Row("POINT (0 1)"),
      Row("LINESTRING (0 0, 0 1, 1 1)"),
      Row("POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))"),
      Row("POLYGON EMPTY"),
      Row("MULTIPOLYGON ( ((0 0, 1 4, 1 0,0 0)) )"),
      Row(null),
      Row("error geometry"),
    )

    val schema = StructType(Array(StructField("geo", StringType, nullable = true)))
    val df = spark.createDataFrame(spark.sparkContext.parallelize(data), schema)
    df.createOrReplaceTempView("data")

    val rst = spark.sql("select ST_ExteriorRing(ST_GeomFromText(geo)) from data ")
    rst.show(false)

    //    rst.queryExecution.debug.codegen()

    val collect = rst.collect()

    assert(collect(0).getAs[GeometryUDT](0).toString == "POINT (0 1)")
    assert(collect(1).getAs[GeometryUDT](0).toString == "LINESTRING (0 0, 0 1, 1 1)")
    assert(collect(2).getAs[GeometryUDT](0).toString == "LINESTRING (0 0, 1 0, 1 1, 0 1, 0 0)")
    assert(collect(3).getAs[GeometryUDT](0).toString == "LINESTRING EMPTY")
    assert(collect(4).getAs[GeometryUDT](0).toString == "MULTIPOLYGON (((0 0, 1 4, 1 0, 0 0)))")
    assert(collect(5).isNullAt(0))
    assert(collect(6).isNullAt(0))

    val rst2 = df.select(st_exteriorring(st_geomfromtext(col("geo"))))
    rst2.show(false)

    val collect2 = rst2.collect()

    assert(collect2(0).getAs[GeometryUDT](0).toString == "POINT (0 1)")
    assert(collect2(1).getAs[GeometryUDT](0).toString == "LINESTRING (0 0, 0 1, 1 1)")
    assert(collect2(2).getAs[GeometryUDT](0).toString == "LINESTRING (0 0, 1 0, 1 1, 0 1, 0 0)")
    assert(collect2(3).getAs[GeometryUDT](0).toString == "LINESTRING EMPTY")
    assert(collect2(4).getAs[GeometryUDT](0).toString == "MULTIPOLYGON (((0 0, 1 4, 1 0, 0 0)))")
    assert(collect2(5).isNullAt(0))
    assert(collect2(6).isNullAt(0))
  }

  test("ST_Scale") {
    val data = Seq(
      Row(GeometryUDT.FromWkt("POINT (120.6 100.999)")),
    )

    val schema = StructType(Array(StructField("geo", new GeometryUDT, nullable = true)))
    val df = spark.createDataFrame(spark.sparkContext.parallelize(data), schema)
    df.createOrReplaceTempView("data")

    val rst = spark.sql("select ST_Scale(geo, 2, 2) from data ")
    rst.show(false)

    //    rst.queryExecution.debug.codegen()

    val collect = rst.collect()

    assert(collect(0).getAs[GeometryUDT](0).toString == "POINT (241.2 201.998)")

    val rst2 = df.select(st_scale(col("geo"), lit(2), lit(2)))
    rst2.show(false)

    val collect2 = rst2.collect()

    assert(collect2(0).getAs[GeometryUDT](0).toString == "POINT (241.2 201.998)")
  }

  test("ST_Scale-Null") {
    val data = Seq(
      Row("POINT (120.6 100.999)"),
      Row(null),
      Row("error geometry"),
    )

    val schema = StructType(Array(StructField("geo", StringType, nullable = true)))
    val df = spark.createDataFrame(spark.sparkContext.parallelize(data), schema)
    df.createOrReplaceTempView("data")

    val rst = spark.sql("select ST_Scale(ST_GeomFromText(geo), 2, 2) from data ")
    rst.show(false)

    //    rst.queryExecution.debug.codegen()

    val collect = rst.collect()

    assert(collect(0).getAs[GeometryUDT](0).toString == "POINT (241.2 201.998)")
    assert(collect(1).isNullAt(0))
    assert(collect(2).isNullAt(0))

    val rst2 = df.select(st_scale(st_geomfromtext(col("geo")), lit(2), lit(2)))
    rst2.show(false)

    val collect2 = rst2.collect()

    assert(collect2(0).getAs[GeometryUDT](0).toString == "POINT (241.2 201.998)")
    assert(collect2(1).isNullAt(0))
    assert(collect2(2).isNullAt(0))
  }

  test("ST_Affine") {
    val data = Seq(
      Row(GeometryUDT.FromWkt("POINT (120.6 100.999)")),
    )

    val schema = StructType(Array(StructField("geo", new GeometryUDT, nullable = true)))
    val df = spark.createDataFrame(spark.sparkContext.parallelize(data), schema)
    df.createOrReplaceTempView("data")

    val rst = spark.sql("select ST_Affine(geo, 2, 2, 2, 2, 2, 2) from data ")
    rst.show(false)

    //    rst.queryExecution.debug.codegen()

    val collect = rst.collect()

    assert(collect(0).getAs[GeometryUDT](0).toString == "POINT (445.198 445.198)")

    val rst2 = df.select(st_affine(col("geo"), lit(2), lit(2), lit(2), lit(2), lit(2), lit(2)))
    rst2.show(false)

    val collect2 = rst2.collect()

    assert(collect2(0).getAs[GeometryUDT](0).toString == "POINT (445.198 445.198)")
  }

  test("ST_Affine-Null") {
    val data = Seq(
      Row("POINT (120.6 100.999)"),
      Row(null),
      Row("error geometry"),
    )

    val schema = StructType(Array(StructField("geo", StringType, nullable = true)))
    val df = spark.createDataFrame(spark.sparkContext.parallelize(data), schema)
    df.createOrReplaceTempView("data")

    val rst = spark.sql("select ST_Affine(ST_GeomFromText(geo), 2, 2, 2, 2, 2, 2) from data ")
    rst.show(false)

    //    rst.queryExecution.debug.codegen()

    val collect = rst.collect()

    assert(collect(0).getAs[GeometryUDT](0).toString == "POINT (445.198 445.198)")
    assert(collect(1).isNullAt(0))
    assert(collect(2).isNullAt(0))

    val rst2 = df.select(st_affine(st_geomfromtext(col("geo")), lit(2), lit(2), lit(2), lit(2), lit(2), lit(2)))
    rst2.show(false)

    val collect2 = rst2.collect()

    assert(collect2(0).getAs[GeometryUDT](0).toString == "POINT (445.198 445.198)")
    assert(collect2(1).isNullAt(0))
    assert(collect2(2).isNullAt(0))
  }
}
