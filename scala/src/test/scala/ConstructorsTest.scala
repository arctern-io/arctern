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

class ConstructorsTest extends AdapterTest {
  test("ST_GeomFromText-Null") {

    val data = Seq(
      Row(1, "POINT (10 20)"),
      Row(2, "LINESTRING (0 0, 10 10, 20 20)"),
      Row(3, "POLYGON ((30 10, 40 40, 20 40, 10 20, 30 10))"),
      Row(4, "MULTIPOINT ((10 40), (40 30), (20 20), (30 10))"),
      Row(5, "MULTIPOLYGON (((30 20, 45 40, 10 40, 30 20)), ((15 5, 40 10, 10 20, 5 10, 15 5)))"),
      Row(6, null),
      Row(7, "plygon(111, 123)")
    )

    val rdd_d = spark.sparkContext.parallelize(data)
    val schema = StructType(Array(StructField("idx", IntegerType, nullable = false), StructField("wkt", StringType, nullable = true)))
    val df = spark.createDataFrame(rdd_d, schema)
    df.createOrReplaceTempView("table_ST_GeomFromText")
    val rst = spark.sql("select idx, ST_GeomFromText(wkt) from table_ST_GeomFromText")

    //    rst.queryExecution.debug.codegen()

    assert(rst.collect()(5).isNullAt(1))
    assert(rst.collect()(6).isNullAt(1))

    rst.show(false)
  }

  test("ST_GeomFromText") {
    val data = Seq(
      Row(1, "POINT (10 20)"),
      Row(2, "LINESTRING (0 0, 10 10, 20 20)"),
      Row(3, "POLYGON ((30 10, 40 40, 20 40, 10 20, 30 10))"),
      Row(4, "MULTIPOINT ((10 40), (40 30), (20 20), (30 10))"),
      Row(5, "MULTIPOLYGON (((30 20, 45 40, 10 40, 30 20)), ((15 5, 40 10, 10 20, 5 10, 15 5)))")
    )

    val rdd_d = spark.sparkContext.parallelize(data)
    val schema = StructType(Array(StructField("idx", IntegerType, nullable = false), StructField("wkt", StringType, nullable = false)))
    val df = spark.createDataFrame(rdd_d, schema)
    df.createOrReplaceTempView("table_ST_GeomFromText")
    val rst = spark.sql("select idx, ST_GeomFromText(wkt) from table_ST_GeomFromText")

    //    rst.queryExecution.debug.codegen()

    rst.show(false)
  }

  test("ST_Point-Null") {

    val data = Seq(
      Row(1.0, 1.1),
      Row(2.1, 2.0),
      Row(3.3, 3.0),
      Row(null, 4.4),
      Row(5.5, 5.5),
      Row(6.6, null),
      Row(null, null)
    )

    val rdd_d = spark.sparkContext.parallelize(data)
    val schema = StructType(Array(StructField("x", DoubleType, nullable = true), StructField("y", DoubleType, nullable = true)))
    val df = spark.createDataFrame(rdd_d, schema)
    df.createOrReplaceTempView("table_ST_Point")
    val rst = spark.sql("select ST_Point(x, y) from table_ST_Point")

    //    rst.queryExecution.debug.codegen()

    assert(rst.collect()(3).isNullAt(0))
    assert(rst.collect()(5).isNullAt(0))
    assert(rst.collect()(6).isNullAt(0))

    rst.show(false)
  }

  test("ST_Point") {
    val data = Seq(
      Row(1.1, 1.1),
      Row(2.1, 2.0),
      Row(3.0, 3.1),
      Row(4.0, 4.0),
    )

    val rdd_d = spark.sparkContext.parallelize(data)
    val schema = StructType(Array(StructField("x", DoubleType, nullable = true), StructField("y", DoubleType, nullable = true)))
    val df = spark.createDataFrame(rdd_d, schema)
    df.createOrReplaceTempView("table_ST_Point")
    val rst = spark.sql("select ST_Point(x, y) from table_ST_Point")

    //    rst.queryExecution.debug.codegen()

    rst.show(false)
  }

  test("ST_PolygonFromEnvelope-Null") {

    val data = Seq(
      Row(1.0, 1.0, 2.0, 2.0),
      Row(null, 1.0, 2.0, 2.0),
      Row(1.0, null, 2.0, 2.0),
      Row(1.0, 1.0, null, 2.0),
      Row(1.0, 1.0, 2.0, null),
      Row(null, null, 2.0, 2.0),
      Row(null, null, null, null),
    )

    val rdd_d = spark.sparkContext.parallelize(data)
    val schema = StructType(Array(StructField("xMin", DoubleType, nullable = true), StructField("yMin", DoubleType, nullable = true), StructField("xMax", DoubleType, nullable = true), StructField("yMax", DoubleType, nullable = true)))
    val df = spark.createDataFrame(rdd_d, schema)
    df.createOrReplaceTempView("table_ST_PolygonFromEnvelope")
    val rst = spark.sql("select ST_PolygonFromEnvelope(xMin, yMin, xMax, yMax) from table_ST_PolygonFromEnvelope")

    //    rst.queryExecution.debug.codegen()

    assert(rst.collect()(1).isNullAt(0))
    assert(rst.collect()(2).isNullAt(0))
    assert(rst.collect()(3).isNullAt(0))
    assert(rst.collect()(4).isNullAt(0))
    assert(rst.collect()(5).isNullAt(0))
    assert(rst.collect()(6).isNullAt(0))

    rst.show(false)
  }

  test("ST_PolygonFromEnvelope") {
    val data = Seq(
      Row(1.0, 1.0, 2.0, 2.0),
      Row(10.1, 10.1, 20.2, 20.2),
      Row(-1.0, -1.0, 2.0, 2.0),
    )

    val rdd_d = spark.sparkContext.parallelize(data)
    val schema = StructType(Array(StructField("xMin", DoubleType, nullable = true), StructField("yMin", DoubleType, nullable = true), StructField("xMax", DoubleType, nullable = true), StructField("yMax", DoubleType, nullable = true)))
    val df = spark.createDataFrame(rdd_d, schema)
    df.createOrReplaceTempView("table_ST_PolygonFromEnvelope")
    val rst = spark.sql("select ST_PolygonFromEnvelope(xMin, yMin, xMax, yMax) from table_ST_PolygonFromEnvelope")

    //    rst.queryExecution.debug.codegen()

    rst.show(false)
  }

  test("ST_GeomFromGeoJSON-Null") {

    val data = Seq(
      Row("""{ "type": "Point", "coordinates": [ 1.0, 2.0 ] }"""),
      Row(null),
      Row("""{ "type": "LineString", "coordinates": [ [ 1.0, 2.0 ], [ 4.0, 5.0 ], [ 7.0, 8.0 ] ] }"""),
      Row(null),
    )

    val rdd_d = spark.sparkContext.parallelize(data)
    val schema = StructType(Array(StructField("json", StringType, nullable = true)))
    val df = spark.createDataFrame(rdd_d, schema)
    df.createOrReplaceTempView("table_ST_GeomFromGeoJSON")
    val rst = spark.sql("select ST_GeomFromGeoJSON(json) from table_ST_GeomFromGeoJSON")

    //    rst.queryExecution.debug.codegen()

    assert(rst.collect()(1).isNullAt(0))
    assert(rst.collect()(3).isNullAt(0))

    rst.show(false)
  }

  test("ST_GeomFromGeoJSON") {
    val data = Seq(
      Row("""{ "type": "Point", "coordinates": [ 1.0, 2.0 ] }"""),
      Row("""{ "type": "LineString", "coordinates": [ [ 1.0, 2.0 ], [ 4.0, 5.0 ], [ 7.0, 8.0 ] ] }"""),
      Row("""{ "type": "Polygon", "coordinates": [ [ [ 0.0, 0.0 ], [ 0.0, 1.0 ], [ 1.0, 1.0 ], [ 1.0, 0.0 ], [ 0.0, 0.0 ] ] ] }"""),
    )

    val rdd_d = spark.sparkContext.parallelize(data)
    val schema = StructType(Array(StructField("json", StringType, nullable = true)))
    val df = spark.createDataFrame(rdd_d, schema)
    df.createOrReplaceTempView("table_ST_GeomFromGeoJSON")
    val rst = spark.sql("select ST_GeomFromGeoJSON(json) from table_ST_GeomFromGeoJSON")

    //    rst.queryExecution.debug.codegen()

    rst.show(false)
  }
}
