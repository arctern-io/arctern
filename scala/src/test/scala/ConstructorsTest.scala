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
import org.apache.spark.sql.functions._
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
    rst.show(false)

    //    rst.queryExecution.debug.codegen()

    val collect = rst.collect()

    assert(collect(0).getAs[GeometryUDT](1).toString == "POINT (10 20)")
    assert(collect(1).getAs[GeometryUDT](1).toString == "LINESTRING (0 0, 10 10, 20 20)")
    assert(collect(2).getAs[GeometryUDT](1).toString == "POLYGON ((30 10, 40 40, 20 40, 10 20, 30 10))")
    assert(collect(3).getAs[GeometryUDT](1).toString == "MULTIPOINT ((10 40), (40 30), (20 20), (30 10))")
    assert(collect(4).getAs[GeometryUDT](1).toString == "MULTIPOLYGON (((30 20, 45 40, 10 40, 30 20)), ((15 5, 40 10, 10 20, 5 10, 15 5)))")
    assert(collect(5).isNullAt(1))
    assert(collect(6).isNullAt(1))

    val rst2 = df.select(st_geomfromtext(col("wkt")))
    rst2.show(false)

    val collect2 = rst2.collect()

    assert(collect2(0).getAs[GeometryUDT](0).toString == "POINT (10 20)")
    assert(collect2(1).getAs[GeometryUDT](0).toString == "LINESTRING (0 0, 10 10, 20 20)")
    assert(collect2(2).getAs[GeometryUDT](0).toString == "POLYGON ((30 10, 40 40, 20 40, 10 20, 30 10))")
    assert(collect2(3).getAs[GeometryUDT](0).toString == "MULTIPOINT ((10 40), (40 30), (20 20), (30 10))")
    assert(collect2(4).getAs[GeometryUDT](0).toString == "MULTIPOLYGON (((30 20, 45 40, 10 40, 30 20)), ((15 5, 40 10, 10 20, 5 10, 15 5)))")
    assert(collect2(5).isNullAt(0))
    assert(collect2(6).isNullAt(0))
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
    rst.show(false)

    //    rst.queryExecution.debug.codegen()

    val collect = rst.collect()

    assert(collect(0).getAs[GeometryUDT](1).toString == "POINT (10 20)")
    assert(collect(1).getAs[GeometryUDT](1).toString == "LINESTRING (0 0, 10 10, 20 20)")
    assert(collect(2).getAs[GeometryUDT](1).toString == "POLYGON ((30 10, 40 40, 20 40, 10 20, 30 10))")
    assert(collect(3).getAs[GeometryUDT](1).toString == "MULTIPOINT ((10 40), (40 30), (20 20), (30 10))")
    assert(collect(4).getAs[GeometryUDT](1).toString == "MULTIPOLYGON (((30 20, 45 40, 10 40, 30 20)), ((15 5, 40 10, 10 20, 5 10, 15 5)))")

    val rst2 = df.select(st_geomfromtext(col("wkt")))
    rst2.show(false)

    val collect2 = rst2.collect()

    assert(collect2(0).getAs[GeometryUDT](0).toString == "POINT (10 20)")
    assert(collect2(1).getAs[GeometryUDT](0).toString == "LINESTRING (0 0, 10 10, 20 20)")
    assert(collect2(2).getAs[GeometryUDT](0).toString == "POLYGON ((30 10, 40 40, 20 40, 10 20, 30 10))")
    assert(collect2(3).getAs[GeometryUDT](0).toString == "MULTIPOINT ((10 40), (40 30), (20 20), (30 10))")
    assert(collect2(4).getAs[GeometryUDT](0).toString == "MULTIPOLYGON (((30 20, 45 40, 10 40, 30 20)), ((15 5, 40 10, 10 20, 5 10, 15 5)))")
  }

  test("ST_GeomFromWKB-Null") {
    val data = Seq(
      Row(1, GeometryUDT.ToWkb(GeometryUDT.FromWkt("POINT (10 20)"))),
      Row(2, GeometryUDT.ToWkb(GeometryUDT.FromWkt("LINESTRING (0 0, 10 10, 20 20)"))),
      Row(3, GeometryUDT.ToWkb(GeometryUDT.FromWkt("POLYGON ((30 10, 40 40, 20 40, 10 20, 30 10))"))),
      Row(4, GeometryUDT.ToWkb(GeometryUDT.FromWkt("MULTIPOINT ((10 40), (40 30), (20 20), (30 10))"))),
      Row(5, GeometryUDT.ToWkb(GeometryUDT.FromWkt("MULTIPOLYGON (((30 20, 45 40, 10 40, 30 20)), ((15 5, 40 10, 10 20, 5 10, 15 5)))"))),
      Row(6, null),
    )

    val rdd_d = spark.sparkContext.parallelize(data)
    val schema = StructType(Array(StructField("idx", IntegerType, nullable = false), StructField("wkb", BinaryType, nullable = true)))
    val df = spark.createDataFrame(rdd_d, schema)
    df.createOrReplaceTempView("table_ST_GeomFromWKB")

    val rst = spark.sql("select idx, ST_GeomFromWKB(wkb) from table_ST_GeomFromWKB")
    rst.show(false)

    //    rst.queryExecution.debug.codegen()

    val collect = rst.collect()

    assert(collect(0).getAs[GeometryUDT](1).toString == "POINT (10 20)")
    assert(collect(1).getAs[GeometryUDT](1).toString == "LINESTRING (0 0, 10 10, 20 20)")
    assert(collect(2).getAs[GeometryUDT](1).toString == "POLYGON ((30 10, 40 40, 20 40, 10 20, 30 10))")
    assert(collect(3).getAs[GeometryUDT](1).toString == "MULTIPOINT ((10 40), (40 30), (20 20), (30 10))")
    assert(collect(4).getAs[GeometryUDT](1).toString == "MULTIPOLYGON (((30 20, 45 40, 10 40, 30 20)), ((15 5, 40 10, 10 20, 5 10, 15 5)))")
    assert(collect(5).isNullAt(1))

    val rst2 = df.select(st_geomfromwkb(col("wkb")))
    rst2.show(false)

    val collect2 = rst2.collect()

    assert(collect2(0).getAs[GeometryUDT](0).toString == "POINT (10 20)")
    assert(collect2(1).getAs[GeometryUDT](0).toString == "LINESTRING (0 0, 10 10, 20 20)")
    assert(collect2(2).getAs[GeometryUDT](0).toString == "POLYGON ((30 10, 40 40, 20 40, 10 20, 30 10))")
    assert(collect2(3).getAs[GeometryUDT](0).toString == "MULTIPOINT ((10 40), (40 30), (20 20), (30 10))")
    assert(collect2(4).getAs[GeometryUDT](0).toString == "MULTIPOLYGON (((30 20, 45 40, 10 40, 30 20)), ((15 5, 40 10, 10 20, 5 10, 15 5)))")
    assert(collect2(5).isNullAt(0))
  }

  test("ST_GeomFromWKB") {
    val data = Seq(
      Row(1, GeometryUDT.ToWkb(GeometryUDT.FromWkt("POINT (10 20)"))),
      Row(2, GeometryUDT.ToWkb(GeometryUDT.FromWkt("LINESTRING (0 0, 10 10, 20 20)"))),
      Row(3, GeometryUDT.ToWkb(GeometryUDT.FromWkt("POLYGON ((30 10, 40 40, 20 40, 10 20, 30 10))"))),
      Row(4, GeometryUDT.ToWkb(GeometryUDT.FromWkt("MULTIPOINT ((10 40), (40 30), (20 20), (30 10))"))),
      Row(5, GeometryUDT.ToWkb(GeometryUDT.FromWkt("MULTIPOLYGON (((30 20, 45 40, 10 40, 30 20)), ((15 5, 40 10, 10 20, 5 10, 15 5)))"))),
    )

    val rdd_d = spark.sparkContext.parallelize(data)
    val schema = StructType(Array(StructField("idx", IntegerType, nullable = false), StructField("wkb", BinaryType, nullable = false)))
    val df = spark.createDataFrame(rdd_d, schema)
    df.createOrReplaceTempView("table_ST_GeomFromWKB")

    val rst = spark.sql("select idx, ST_GeomFromWKB(wkb) from table_ST_GeomFromWKB")
    rst.show(false)

    //    rst.queryExecution.debug.codegen()

    val collect = rst.collect()

    assert(collect(0).getAs[GeometryUDT](1).toString == "POINT (10 20)")
    assert(collect(1).getAs[GeometryUDT](1).toString == "LINESTRING (0 0, 10 10, 20 20)")
    assert(collect(2).getAs[GeometryUDT](1).toString == "POLYGON ((30 10, 40 40, 20 40, 10 20, 30 10))")
    assert(collect(3).getAs[GeometryUDT](1).toString == "MULTIPOINT ((10 40), (40 30), (20 20), (30 10))")
    assert(collect(4).getAs[GeometryUDT](1).toString == "MULTIPOLYGON (((30 20, 45 40, 10 40, 30 20)), ((15 5, 40 10, 10 20, 5 10, 15 5)))")

    val rst2 = df.select(st_geomfromwkb(col("wkb")))
    rst2.show(false)

    val collect2 = rst2.collect()

    assert(collect2(0).getAs[GeometryUDT](0).toString == "POINT (10 20)")
    assert(collect2(1).getAs[GeometryUDT](0).toString == "LINESTRING (0 0, 10 10, 20 20)")
    assert(collect2(2).getAs[GeometryUDT](0).toString == "POLYGON ((30 10, 40 40, 20 40, 10 20, 30 10))")
    assert(collect2(3).getAs[GeometryUDT](0).toString == "MULTIPOINT ((10 40), (40 30), (20 20), (30 10))")
    assert(collect2(4).getAs[GeometryUDT](0).toString == "MULTIPOLYGON (((30 20, 45 40, 10 40, 30 20)), ((15 5, 40 10, 10 20, 5 10, 15 5)))")
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
    rst.show(false)

    //    rst.queryExecution.debug.codegen()

    val collect = rst.collect()

    assert(collect(0).getAs[GeometryUDT](0).toString == "POINT (1 1.1)")
    assert(collect(1).getAs[GeometryUDT](0).toString == "POINT (2.1 2)")
    assert(collect(2).getAs[GeometryUDT](0).toString == "POINT (3.3 3)")
    assert(collect(3).isNullAt(0))
    assert(collect(4).getAs[GeometryUDT](0).toString == "POINT (5.5 5.5)")
    assert(collect(5).isNullAt(0))
    assert(collect(6).isNullAt(0))

    val rst2 = df.select(st_point(col("x"), col("y")))
    rst2.show(false)

    val collect2 = rst2.collect()

    assert(collect2(0).getAs[GeometryUDT](0).toString == "POINT (1 1.1)")
    assert(collect2(1).getAs[GeometryUDT](0).toString == "POINT (2.1 2)")
    assert(collect2(2).getAs[GeometryUDT](0).toString == "POINT (3.3 3)")
    assert(collect2(3).isNullAt(0))
    assert(collect2(4).getAs[GeometryUDT](0).toString == "POINT (5.5 5.5)")
    assert(collect2(5).isNullAt(0))
    assert(collect2(6).isNullAt(0))
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
    rst.show(false)

    //    rst.queryExecution.debug.codegen()

    val collect = rst.collect()

    assert(collect(0).getAs[GeometryUDT](0).toString == "POINT (1.1 1.1)")
    assert(collect(1).getAs[GeometryUDT](0).toString == "POINT (2.1 2)")
    assert(collect(2).getAs[GeometryUDT](0).toString == "POINT (3 3.1)")
    assert(collect(3).getAs[GeometryUDT](0).toString == "POINT (4 4)")

    val rst2 = df.select(st_point(col("x"), col("y")))
    rst2.show(false)

    val collect2 = rst2.collect()

    assert(collect2(0).getAs[GeometryUDT](0).toString == "POINT (1.1 1.1)")
    assert(collect2(1).getAs[GeometryUDT](0).toString == "POINT (2.1 2)")
    assert(collect2(2).getAs[GeometryUDT](0).toString == "POINT (3 3.1)")
    assert(collect2(3).getAs[GeometryUDT](0).toString == "POINT (4 4)")
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
    rst.show(false)

    //    rst.queryExecution.debug.codegen()

    val collect = rst.collect()

    assert(collect(0).getAs[GeometryUDT](0).toString == "POLYGON ((1 1, 1 2, 2 2, 2 1, 1 1))")
    assert(collect(1).isNullAt(0))
    assert(collect(2).isNullAt(0))
    assert(collect(3).isNullAt(0))
    assert(collect(4).isNullAt(0))
    assert(collect(5).isNullAt(0))
    assert(collect(6).isNullAt(0))

    val rst2 = df.select(st_polygonfromenvelope(col("xMin"), col("yMin"), col("xMax"), col("yMax")))
    rst2.show(false)

    val collect2 = rst2.collect()

    assert(collect2(0).getAs[GeometryUDT](0).toString == "POLYGON ((1 1, 1 2, 2 2, 2 1, 1 1))")
    assert(collect2(1).isNullAt(0))
    assert(collect2(2).isNullAt(0))
    assert(collect2(3).isNullAt(0))
    assert(collect2(4).isNullAt(0))
    assert(collect2(5).isNullAt(0))
    assert(collect2(6).isNullAt(0))
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
    rst.show(false)

    //    rst.queryExecution.debug.codegen()

    val collect = rst.collect()

    assert(collect(0).getAs[GeometryUDT](0).toString == "POLYGON ((1 1, 1 2, 2 2, 2 1, 1 1))")
    assert(collect(1).getAs[GeometryUDT](0).toString == "POLYGON ((10.1 10.1, 10.1 20.2, 20.2 20.2, 20.2 10.1, 10.1 10.1))")
    assert(collect(2).getAs[GeometryUDT](0).toString == "POLYGON ((-1 -1, -1 2, 2 2, 2 -1, -1 -1))")

    val rst2 = df.select(st_polygonfromenvelope(col("xMin"), col("yMin"), col("xMax"), col("yMax")))
    rst2.show(false)

    val collect2 = rst2.collect()

    assert(collect2(0).getAs[GeometryUDT](0).toString == "POLYGON ((1 1, 1 2, 2 2, 2 1, 1 1))")
    assert(collect2(1).getAs[GeometryUDT](0).toString == "POLYGON ((10.1 10.1, 10.1 20.2, 20.2 20.2, 20.2 10.1, 10.1 10.1))")
    assert(collect2(2).getAs[GeometryUDT](0).toString == "POLYGON ((-1 -1, -1 2, 2 2, 2 -1, -1 -1))")
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
    rst.show(false)

    //    rst.queryExecution.debug.codegen()

    val collect = rst.collect()

    assert(collect(0).getAs[GeometryUDT](0).toString == "POINT (1 2)")
    assert(collect(1).isNullAt(0))
    assert(collect(2).getAs[GeometryUDT](0).toString == "LINESTRING (1 2, 4 5, 7 8)")
    assert(collect(3).isNullAt(0))

    val rst2 = df.select(st_geomfromgeojson(col("json")))
    rst2.show(false)

    val collect2 = rst2.collect()

    assert(collect2(0).getAs[GeometryUDT](0).toString == "POINT (1 2)")
    assert(collect2(1).isNullAt(0))
    assert(collect2(2).getAs[GeometryUDT](0).toString == "LINESTRING (1 2, 4 5, 7 8)")
    assert(collect2(3).isNullAt(0))
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
    rst.show(false)

    //    rst.queryExecution.debug.codegen()

    val collect = rst.collect()

    assert(collect(0).getAs[GeometryUDT](0).toString == "POINT (1 2)")
    assert(collect(1).getAs[GeometryUDT](0).toString == "LINESTRING (1 2, 4 5, 7 8)")
    assert(collect(2).getAs[GeometryUDT](0).toString == "POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))")

    val rst2 = df.select(st_geomfromgeojson(col("json")))
    rst2.show(false)

    val collect2 = rst2.collect()

    assert(collect2(0).getAs[GeometryUDT](0).toString == "POINT (1 2)")
    assert(collect2(1).getAs[GeometryUDT](0).toString == "LINESTRING (1 2, 4 5, 7 8)")
    assert(collect2(2).getAs[GeometryUDT](0).toString == "POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))")
  }

  test("ST_AsText-Null") {
    val data = Seq(
      Row("POINT (10 20)"),
      Row(null),
      Row("POLYGON ((30 10, 40 40, 20 40, 10 20, 30 10))"),
      Row(null),
      Row("MULTIPOLYGON (((30 20, 45 40, 10 40, 30 20)), ((15 5, 40 10, 10 20, 5 10, 15 5)))"),
    )

    val rdd_d = spark.sparkContext.parallelize(data)
    val schema = StructType(Array(StructField("wkt", StringType, nullable = true)))
    val df = spark.createDataFrame(rdd_d, schema)
    df.createOrReplaceTempView("table_ST_AsText")

    val rst = spark.sql("select ST_AsText(ST_GeomFromText(wkt)) from table_ST_AsText")
    rst.show(false)

    //    rst.queryExecution.debug.codegen()

    val collect = rst.collect()

    assert(collect(0).getAs[GeometryUDT](0).toString == "POINT (10 20)")
    assert(collect(1).isNullAt(0))
    assert(collect(2).getAs[GeometryUDT](0).toString == "POLYGON ((30 10, 40 40, 20 40, 10 20, 30 10))")
    assert(collect(3).isNullAt(0))
    assert(collect(4).getAs[GeometryUDT](0).toString == "MULTIPOLYGON (((30 20, 45 40, 10 40, 30 20)), ((15 5, 40 10, 10 20, 5 10, 15 5)))")

    val rst2 = df.select(st_astext(st_geomfromtext(col("wkt"))))
    rst2.show(false)

    val collect2 = rst2.collect()

    assert(collect2(0).getAs[GeometryUDT](0).toString == "POINT (10 20)")
    assert(collect2(1).isNullAt(0))
    assert(collect2(2).getAs[GeometryUDT](0).toString == "POLYGON ((30 10, 40 40, 20 40, 10 20, 30 10))")
    assert(collect2(3).isNullAt(0))
    assert(collect2(4).getAs[GeometryUDT](0).toString == "MULTIPOLYGON (((30 20, 45 40, 10 40, 30 20)), ((15 5, 40 10, 10 20, 5 10, 15 5)))")
  }

  test("ST_AsText-FromArcternExpr") {
    val data = Seq(
      Row("POINT (10 20)"),
      Row("LINESTRING (0 0, 10 10, 20 20)"),
      Row("POLYGON ((30 10, 40 40, 20 40, 10 20, 30 10))"),
      Row("MULTIPOINT ((10 40), (40 30), (20 20), (30 10))"),
      Row("MULTIPOLYGON (((30 20, 45 40, 10 40, 30 20)), ((15 5, 40 10, 10 20, 5 10, 15 5)))"),
    )

    val rdd_d = spark.sparkContext.parallelize(data)
    val schema = StructType(Array(StructField("wkt", StringType, nullable = true)))
    val df = spark.createDataFrame(rdd_d, schema)
    df.createOrReplaceTempView("table_ST_AsText")

    val rst = spark.sql("select ST_AsText(ST_GeomFromText(wkt)) from table_ST_AsText")
    rst.show(false)

    //    rst.queryExecution.debug.codegen()

    val collect = rst.collect()

    assert(collect(0).getAs[GeometryUDT](0).toString == "POINT (10 20)")
    assert(collect(1).getAs[GeometryUDT](0).toString == "LINESTRING (0 0, 10 10, 20 20)")
    assert(collect(2).getAs[GeometryUDT](0).toString == "POLYGON ((30 10, 40 40, 20 40, 10 20, 30 10))")
    assert(collect(3).getAs[GeometryUDT](0).toString == "MULTIPOINT ((10 40), (40 30), (20 20), (30 10))")
    assert(collect(4).getAs[GeometryUDT](0).toString == "MULTIPOLYGON (((30 20, 45 40, 10 40, 30 20)), ((15 5, 40 10, 10 20, 5 10, 15 5)))")

    val rst2 = df.select(st_astext(st_geomfromtext(col("wkt"))))
    rst2.show(false)

    val collect2 = rst2.collect()

    assert(collect2(0).getAs[GeometryUDT](0).toString == "POINT (10 20)")
    assert(collect2(1).getAs[GeometryUDT](0).toString == "LINESTRING (0 0, 10 10, 20 20)")
    assert(collect2(2).getAs[GeometryUDT](0).toString == "POLYGON ((30 10, 40 40, 20 40, 10 20, 30 10))")
    assert(collect2(3).getAs[GeometryUDT](0).toString == "MULTIPOINT ((10 40), (40 30), (20 20), (30 10))")
    assert(collect2(4).getAs[GeometryUDT](0).toString == "MULTIPOLYGON (((30 20, 45 40, 10 40, 30 20)), ((15 5, 40 10, 10 20, 5 10, 15 5)))")
  }

  test("ST_AsText-FromNormalExpr") {
    val data = Seq(
      Row(GeometryUDT.FromWkt("POINT (10 20)")),
      Row(GeometryUDT.FromWkt("LINESTRING (0 0, 10 10, 20 20)")),
      Row(GeometryUDT.FromWkt("POLYGON ((30 10, 40 40, 20 40, 10 20, 30 10))")),
      Row(GeometryUDT.FromWkt("MULTIPOINT ((10 40), (40 30), (20 20), (30 10))")),
      Row(GeometryUDT.FromWkt("MULTIPOLYGON (((30 20, 45 40, 10 40, 30 20)), ((15 5, 40 10, 10 20, 5 10, 15 5)))")),
      Row(null)
    )

    val rdd_d = spark.sparkContext.parallelize(data)
    val schema = StructType(Array(StructField("geo", new GeometryUDT, nullable = true)))
    val df = spark.createDataFrame(rdd_d, schema)
    df.createOrReplaceTempView("table_ST_AsText")

    val rst = spark.sql("select ST_AsText(geo) from table_ST_AsText")
    rst.show(false)

    //    rst.queryExecution.debug.codegen()

    val collect = rst.collect()

    assert(collect(0).getAs[GeometryUDT](0).toString == "POINT (10 20)")
    assert(collect(1).getAs[GeometryUDT](0).toString == "LINESTRING (0 0, 10 10, 20 20)")
    assert(collect(2).getAs[GeometryUDT](0).toString == "POLYGON ((30 10, 40 40, 20 40, 10 20, 30 10))")
    assert(collect(3).getAs[GeometryUDT](0).toString == "MULTIPOINT ((10 40), (40 30), (20 20), (30 10))")
    assert(collect(4).getAs[GeometryUDT](0).toString == "MULTIPOLYGON (((30 20, 45 40, 10 40, 30 20)), ((15 5, 40 10, 10 20, 5 10, 15 5)))")
    assert(collect(5).isNullAt(0))

    val rst2 = df.select(st_astext(col("geo")))
    rst2.show(false)

    val collect2 = rst2.collect()

    assert(collect2(0).getAs[GeometryUDT](0).toString == "POINT (10 20)")
    assert(collect2(1).getAs[GeometryUDT](0).toString == "LINESTRING (0 0, 10 10, 20 20)")
    assert(collect2(2).getAs[GeometryUDT](0).toString == "POLYGON ((30 10, 40 40, 20 40, 10 20, 30 10))")
    assert(collect2(3).getAs[GeometryUDT](0).toString == "MULTIPOINT ((10 40), (40 30), (20 20), (30 10))")
    assert(collect2(4).getAs[GeometryUDT](0).toString == "MULTIPOLYGON (((30 20, 45 40, 10 40, 30 20)), ((15 5, 40 10, 10 20, 5 10, 15 5)))")
    assert(collect2(5).isNullAt(0))
  }

  test("ST_AsGeoJSON-Null") {
    val data = Seq(
      Row("POINT (10 20)"),
      Row(null),
      Row("POLYGON ((30 10, 40 40, 20 40, 10 20, 30 10))"),
      Row(null),
      Row("MULTIPOLYGON (((30 20, 45 40, 10 40, 30 20)), ((15 5, 40 10, 10 20, 5 10, 15 5)))"),
    )

    val rdd_d = spark.sparkContext.parallelize(data)
    val schema = StructType(Array(StructField("wkt", StringType, nullable = true)))
    val df = spark.createDataFrame(rdd_d, schema)
    df.createOrReplaceTempView("table_ST_AsGeoJSON")

    val rst = spark.sql("select ST_AsGeoJSON(ST_GeomFromText(wkt)) from table_ST_AsGeoJSON")
    rst.show(false)

    //    rst.queryExecution.debug.codegen()

    val collect = rst.collect()

    assert(collect(0).getAs[GeometryUDT](0).toString == """{"type":"Point","coordinates":[10.0,20.0]}""")
    assert(collect(1).isNullAt(0))
    assert(collect(2).getAs[GeometryUDT](0).toString == """{"type":"Polygon","coordinates":[[[30.0,10.0],[40.0,40.0],[20.0,40.0],[10.0,20.0],[30.0,10.0]]]}""")
    assert(collect(3).isNullAt(0))
    assert(collect(4).getAs[GeometryUDT](0).toString == """{"type":"MultiPolygon","coordinates":[[[[30.0,20.0],[45.0,40.0],[10.0,40.0],[30.0,20.0]]],[[[15.0,5.0],[40.0,10.0],[10.0,20.0],[5.0,10.0],[15.0,5.0]]]]}""")

    val rst2 = df.select(st_asgeojson(st_geomfromtext(col("wkt"))))
    rst2.show(false)

    val collect2 = rst2.collect()

    assert(collect2(0).getAs[GeometryUDT](0).toString == """{"type":"Point","coordinates":[10.0,20.0]}""")
    assert(collect2(1).isNullAt(0))
    assert(collect2(2).getAs[GeometryUDT](0).toString == """{"type":"Polygon","coordinates":[[[30.0,10.0],[40.0,40.0],[20.0,40.0],[10.0,20.0],[30.0,10.0]]]}""")
    assert(collect2(3).isNullAt(0))
    assert(collect2(4).getAs[GeometryUDT](0).toString == """{"type":"MultiPolygon","coordinates":[[[[30.0,20.0],[45.0,40.0],[10.0,40.0],[30.0,20.0]]],[[[15.0,5.0],[40.0,10.0],[10.0,20.0],[5.0,10.0],[15.0,5.0]]]]}""")
  }

  test("ST_AsGeoJSON-FromArcternExpr") {
    val data = Seq(
      Row("POINT (10 20)"),
      Row("LINESTRING (0 0, 10 10, 20 20)"),
      Row("POLYGON ((30 10, 40 40, 20 40, 10 20, 30 10))"),
      Row("MULTIPOINT ((10 40), (40 30), (20 20), (30 10))"),
      Row("MULTIPOLYGON (((30 20, 45 40, 10 40, 30 20)), ((15 5, 40 10, 10 20, 5 10, 15 5)))"),
    )

    val rdd_d = spark.sparkContext.parallelize(data)
    val schema = StructType(Array(StructField("wkt", StringType, nullable = true)))
    val df = spark.createDataFrame(rdd_d, schema)
    df.createOrReplaceTempView("table_ST_AsGeoJSON")

    val rst = spark.sql("select ST_AsGeoJSON(ST_GeomFromText(wkt)) from table_ST_AsGeoJSON")
    rst.show(false)

    //    rst.queryExecution.debug.codegen()

    val collect = rst.collect()

    assert(collect(0).getAs[GeometryUDT](0).toString == """{"type":"Point","coordinates":[10.0,20.0]}""")
    assert(collect(1).getAs[GeometryUDT](0).toString == """{"type":"LineString","coordinates":[[0.0,0.0],[10.0,10.0],[20.0,20.0]]}""")
    assert(collect(2).getAs[GeometryUDT](0).toString == """{"type":"Polygon","coordinates":[[[30.0,10.0],[40.0,40.0],[20.0,40.0],[10.0,20.0],[30.0,10.0]]]}""")
    assert(collect(3).getAs[GeometryUDT](0).toString == """{"type":"MultiPoint","coordinates":[[10.0,40.0],[40.0,30.0],[20.0,20.0],[30.0,10.0]]}""")
    assert(collect(4).getAs[GeometryUDT](0).toString == """{"type":"MultiPolygon","coordinates":[[[[30.0,20.0],[45.0,40.0],[10.0,40.0],[30.0,20.0]]],[[[15.0,5.0],[40.0,10.0],[10.0,20.0],[5.0,10.0],[15.0,5.0]]]]}""")

    val rst2 = df.select(st_asgeojson(st_geomfromtext(col("wkt"))))
    rst2.show(false)

    val collect2 = rst2.collect()

    assert(collect2(0).getAs[GeometryUDT](0).toString == """{"type":"Point","coordinates":[10.0,20.0]}""")
    assert(collect2(1).getAs[GeometryUDT](0).toString == """{"type":"LineString","coordinates":[[0.0,0.0],[10.0,10.0],[20.0,20.0]]}""")
    assert(collect2(2).getAs[GeometryUDT](0).toString == """{"type":"Polygon","coordinates":[[[30.0,10.0],[40.0,40.0],[20.0,40.0],[10.0,20.0],[30.0,10.0]]]}""")
    assert(collect2(3).getAs[GeometryUDT](0).toString == """{"type":"MultiPoint","coordinates":[[10.0,40.0],[40.0,30.0],[20.0,20.0],[30.0,10.0]]}""")
    assert(collect2(4).getAs[GeometryUDT](0).toString == """{"type":"MultiPolygon","coordinates":[[[[30.0,20.0],[45.0,40.0],[10.0,40.0],[30.0,20.0]]],[[[15.0,5.0],[40.0,10.0],[10.0,20.0],[5.0,10.0],[15.0,5.0]]]]}""")
  }

  test("ST_AsGeoJSON-FromNormalExpr") {
    val data = Seq(
      Row(GeometryUDT.FromWkt("POINT (10 20)")),
      Row(GeometryUDT.FromWkt("LINESTRING (0 0, 10 10, 20 20)")),
      Row(GeometryUDT.FromWkt("POLYGON ((30 10, 40 40, 20 40, 10 20, 30 10))")),
      Row(GeometryUDT.FromWkt("MULTIPOINT ((10 40), (40 30), (20 20), (30 10))")),
      Row(GeometryUDT.FromWkt("MULTIPOLYGON (((30 20, 45 40, 10 40, 30 20)), ((15 5, 40 10, 10 20, 5 10, 15 5)))")),
      Row(null)
    )

    val rdd_d = spark.sparkContext.parallelize(data)
    val schema = StructType(Array(StructField("geo", new GeometryUDT, nullable = true)))
    val df = spark.createDataFrame(rdd_d, schema)
    df.createOrReplaceTempView("table_ST_AsGeoJSON")

    val rst = spark.sql("select ST_AsGeoJSON(geo) from table_ST_AsGeoJSON")
    rst.show(false)

    //    rst.queryExecution.debug.codegen()

    val collect = rst.collect()

    assert(collect(0).getAs[GeometryUDT](0).toString == """{"type":"Point","coordinates":[10.0,20.0]}""")
    assert(collect(1).getAs[GeometryUDT](0).toString == """{"type":"LineString","coordinates":[[0.0,0.0],[10.0,10.0],[20.0,20.0]]}""")
    assert(collect(2).getAs[GeometryUDT](0).toString == """{"type":"Polygon","coordinates":[[[30.0,10.0],[40.0,40.0],[20.0,40.0],[10.0,20.0],[30.0,10.0]]]}""")
    assert(collect(3).getAs[GeometryUDT](0).toString == """{"type":"MultiPoint","coordinates":[[10.0,40.0],[40.0,30.0],[20.0,20.0],[30.0,10.0]]}""")
    assert(collect(4).getAs[GeometryUDT](0).toString == """{"type":"MultiPolygon","coordinates":[[[[30.0,20.0],[45.0,40.0],[10.0,40.0],[30.0,20.0]]],[[[15.0,5.0],[40.0,10.0],[10.0,20.0],[5.0,10.0],[15.0,5.0]]]]}""")
    assert(collect(5).isNullAt(0))

    val rst2 = df.select(st_asgeojson(col("geo")))
    rst2.show(false)

    val collect2 = rst2.collect()

    assert(collect2(0).getAs[GeometryUDT](0).toString == """{"type":"Point","coordinates":[10.0,20.0]}""")
    assert(collect2(1).getAs[GeometryUDT](0).toString == """{"type":"LineString","coordinates":[[0.0,0.0],[10.0,10.0],[20.0,20.0]]}""")
    assert(collect2(2).getAs[GeometryUDT](0).toString == """{"type":"Polygon","coordinates":[[[30.0,10.0],[40.0,40.0],[20.0,40.0],[10.0,20.0],[30.0,10.0]]]}""")
    assert(collect2(3).getAs[GeometryUDT](0).toString == """{"type":"MultiPoint","coordinates":[[10.0,40.0],[40.0,30.0],[20.0,20.0],[30.0,10.0]]}""")
    assert(collect2(4).getAs[GeometryUDT](0).toString == """{"type":"MultiPolygon","coordinates":[[[[30.0,20.0],[45.0,40.0],[10.0,40.0],[30.0,20.0]]],[[[15.0,5.0],[40.0,10.0],[10.0,20.0],[5.0,10.0],[15.0,5.0]]]]}""")
    assert(collect2(5).isNullAt(0))
  }
}
