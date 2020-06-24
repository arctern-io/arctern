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
import org.apache.spark.sql.arctern.expressions.utils.arcternUnion
import org.locationtech.jts.io.WKTReader

class AggregateFunctionsTest extends AdapterTest {
  test("ST_Union_Aggr") {
    val data = Seq(
      Row(GeometryUDT.FromWkt("POINT (-1 -1)")),
      Row(GeometryUDT.FromWkt("POINT (5 5)")),
      Row(GeometryUDT.FromWkt("LINESTRING (0 0, 10 10, 20 20)")),
      Row(GeometryUDT.FromWkt("POLYGON ((1 1,2 1,2 2,1 2,1 1))")),
      Row(GeometryUDT.FromWkt("MULTIPOLYGON ( ((0 0, 1 0, 1 1, 0 1, 0 0)) )")),
    )

    val schema = StructType(Array(StructField("geo", new GeometryUDT, nullable = true)))
    val df = spark.createDataFrame(spark.sparkContext.parallelize(data), schema)
    df.createOrReplaceTempView("ST_Union_Aggr_data")

    val rst = spark.sql("select ST_Union_Aggr(geo) from ST_Union_Aggr_data")
    rst.show(false)

    val collect = rst.collect()

    assert(GeometryUDT.FromWkt(collect(0).getAs[GeometryUDT](0).toString).getArea==2.0)

    val rst2 = df.agg(st_union_aggr(col("geo")))
    rst2.show(false)

    val collect2 = rst2.collect()

    assert(GeometryUDT.FromWkt(collect2(0).getAs[GeometryUDT](0).toString).getArea==2.0)
  }

  test("ST_Union_Aggr-Null") {
    val data = Seq(
      Row("error geo"),
      Row("POINT (-1 -1)"),
      Row("LINESTRING (0 0, 10 10, 20 20)"),
      Row("POLYGON ((1 1,2 1,2 2,1 2,1 1))"),
      Row("MULTIPOLYGON ( ((0 0, 1 0, 1 1, 0 1, 0 0)) )"),
      Row(null),
    )

    val schema = StructType(Array(StructField("geo", StringType, nullable = true)))
    val df = spark.createDataFrame(spark.sparkContext.parallelize(data), schema)
    df.createOrReplaceTempView("ST_Union_Aggr_Null_data")

    val rst = spark.sql("select ST_Union_Aggr(ST_GeomFromText(geo)) from ST_Union_Aggr_Null_data")
    rst.show(false)

    val collect = rst.collect()

    assert(GeometryUDT.FromWkt(collect(0).getAs[GeometryUDT](0).toString).getArea==2.0)

    val rst2 = df.agg(st_union_aggr(st_geomfromtext(col("geo"))))
    rst2.show(false)

    val collect2 = rst2.collect()

    assert(GeometryUDT.FromWkt(collect2(0).getAs[GeometryUDT](0).toString).getArea==2.0)
  }

  test("arcternUnion") {
    val g0 = new WKTReader().read("GEOMETRYCOLLECTION (POINT (5 5), POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0)))")
    val g1 = new WKTReader().read("POLYGON ((10 10, 20 10, 20 20, 10 20, 10 10))")
    val g2 = new WKTReader().read("POINT (10 10)")

    val res0 = arcternUnion(g0, g1)
    val res1 = arcternUnion(g0, g2)
    val res2 = arcternUnion(g1, g2)

    println(res0)
    println(res1)
    println(res2)

    assert(res0.toString.equals("GEOMETRYCOLLECTION (POINT (5 5), MULTIPOLYGON (((0 0, 0 1, 1 1, 1 0, 0 0)), ((10 10, 10 20, 20 20, 20 10, 10 10))))"))
    assert(res1.toString.equals("GEOMETRYCOLLECTION (MULTIPOINT ((5 5), (10 10)), POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0)))"))
    assert(res2.toString.equals("POLYGON ((10 10, 10 20, 20 20, 20 10, 10 10))"))
  }

  test("ST_Envelope_Aggr") {
    val data = Seq(
      Row(GeometryUDT.FromWkt("POINT (-1 -1)")),
      Row(GeometryUDT.FromWkt("LINESTRING (0 0, 10 10, 20 20)")),
      Row(GeometryUDT.FromWkt("POLYGON ((1 1,2 1,2 2,2 1,1 1))")),
      Row(GeometryUDT.FromWkt("MULTIPOLYGON ( ((0 0, 1 0, 1 1, 0 1, 0 0)) )")),
    )

    val schema = StructType(Array(StructField("geo", new GeometryUDT, nullable = true)))
    val df = spark.createDataFrame(spark.sparkContext.parallelize(data), schema)
    df.createOrReplaceTempView("ST_Envelope_Aggr_data")

    val rst = spark.sql("select ST_Envelope_Aggr(geo) from ST_Envelope_Aggr_data")
    rst.show(false)

    val collect = rst.collect()

    assert(collect(0).getAs[GeometryUDT](0).toString == "POLYGON ((-1 -1, -1 20, 20 20, 20 -1, -1 -1))")

    val rst2 = df.agg(st_envelope_aggr(col("geo")))
    rst2.show(false)

    val collect2 = rst2.collect()

    assert(collect2(0).getAs[GeometryUDT](0).toString == "POLYGON ((-1 -1, -1 20, 20 20, 20 -1, -1 -1))")
  }

  test("ST_Envelope_Aggr-Null") {
    val data = Seq(
      Row("error geo"),
      Row("POINT (0 0)"),
      Row("LINESTRING (0 0, 10 10, 20 20)"),
      Row(null),
      Row("POLYGON ((1 1,2 1,2 2,2 1,1 1))"),
      Row("MULTIPOLYGON ( ((0 0, 1 0, 1 1, 0 1, 0 0)) )"),
    )

    val schema = StructType(Array(StructField("geo", StringType, nullable = true)))
    val df = spark.createDataFrame(spark.sparkContext.parallelize(data), schema)
    df.createOrReplaceTempView("ST_Envelope_Aggr_Null_data")

    val rst = spark.sql("select ST_Envelope_Aggr(ST_GeomFromText(geo)) from ST_Envelope_Aggr_Null_data")
    rst.show(false)

    val collect = rst.collect()

    assert(collect(0).getAs[GeometryUDT](0).toString == "POLYGON ((0 0, 0 20, 20 20, 20 0, 0 0))")

    val rst2 = df.agg(st_envelope_aggr(st_geomfromtext(col("geo"))))
    rst2.show(false)

    val collect2 = rst2.collect()

    assert(collect2(0).getAs[GeometryUDT](0).toString == "POLYGON ((0 0, 0 20, 20 20, 20 0, 0 0))")
  }
}
