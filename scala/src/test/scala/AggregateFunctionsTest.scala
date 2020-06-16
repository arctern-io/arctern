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
import org.apache.spark.sql.types._

class AggregateFunctionsTest extends AdapterTest {
  test("ST_Union_Aggr") {
    val data = Seq(
      Row(GeometryUDT.FromWkt("POINT (0 0)")),
      Row(GeometryUDT.FromWkt("LINESTRING (0 0, 10 10, 20 20)")),
      Row(GeometryUDT.FromWkt("POLYGON ((1 1,2 1,2 2,2 1,1 1))")),
      Row(GeometryUDT.FromWkt("MULTIPOLYGON ( ((0 0, 1 0, 1 1, 0 1, 0 0)) )")),
    )

    val schema = StructType(Array(StructField("geo", new GeometryUDT, nullable = true)))
    val df = spark.createDataFrame(spark.sparkContext.parallelize(data), schema)
    df.createOrReplaceTempView("ST_Union_Aggr_data")

    val rst = spark.sql("select ST_Union_Aggr(geo) from ST_Union_Aggr_data")

    rst.show(false)
  }

  test("ST_Union_Aggr-Null") {
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
    df.createOrReplaceTempView("ST_Union_Aggr_Null_data")

    val rst = spark.sql("select ST_Union_Aggr(ST_GeomFromText(geo)) from ST_Union_Aggr_Null_data")
    val collect = rst.collect()

    assert(!collect(0).isNullAt(0))

    rst.show(false)
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
    val collect = rst.collect()

    assert(!collect(0).isNullAt(0))

    rst.show(false)
  }

}
