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

class AggregateFunctionsTest extends AdapterTest {
  test("ST_Envelope_Aggr_codegen") {
    assert(true)
    val data = Seq(
      Row(GeometryUDT.FromWkt("Polygon((0 0, 0 1, 1 1, 1 0, 0 0))")),
      Row(GeometryUDT.FromWkt("LINESTRING (0 0, 10 10, 20 20)")),
      Row(GeometryUDT.FromWkt("POINT (0 0)")),
      Row(GeometryUDT.FromWkt("POLYGON ((1 2,3 4,5 6,1 2))")),
      Row(GeometryUDT.FromWkt("MULTIPOLYGON ( ((0 0, 1 0, 1 1,0 0)) )")),
    )

    val schema = StructType(Array(StructField("geo", new GeometryUDT, nullable = true)))
    val df = spark.createDataFrame(spark.sparkContext.parallelize(data), schema)
    df.createOrReplaceTempView("raw_data")

    df.show(false)
    val rst = spark.sql("select ST_Envelope_Aggr(geo) from raw_data")
    //    rst.queryExecution.debug.codegen()
    rst.show(false)
    val collect = rst.collect()
    assert(collect(0).getAs[GeometryUDT](0).toString == "POLYGON ((0 0, 0 20, 20 20, 20 0, 0 0))")
  }

  test("ST_Union_Aggr") {
    val data = Seq(
      Row(GeometryUDT.FromWkt("POLYGON EMPTY")),
      Row(GeometryUDT.FromWkt("POLYGON EMPTY")),
      Row(GeometryUDT.FromWkt("POINT (-1 -1)")),
      Row(GeometryUDT.FromWkt("POINT (5 5)")),
      Row(GeometryUDT.FromWkt("POINT (1 1)")),
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

    assert(GeometryUDT.FromWkt(collect(0).getAs[GeometryUDT](0).toString).getArea == 2.0)

    val rst2 = df.agg(st_union_aggr(col("geo")))
    rst2.show(false)

    val collect2 = rst2.collect()

    assert(GeometryUDT.FromWkt(collect2(0).getAs[GeometryUDT](0).toString).getArea == 2.0)
  }

  test("ST_Union_Aggr-SimpleTest") {
    val data = Seq(
      Row(GeometryUDT.FromWkt("POINT (5 5)")),
      Row(GeometryUDT.FromWkt("POLYGON ((1 1,2 1,2 2,1 2,1 1))")),
    )

    val schema = StructType(Array(StructField("geo", new GeometryUDT, nullable = true)))
    val df = spark.createDataFrame(spark.sparkContext.parallelize(data), schema)
    df.createOrReplaceTempView("ST_Union_Aggr_data")

    val rst = spark.sql("select ST_Union_Aggr(geo) from ST_Union_Aggr_data")
    rst.show(false)

    val collect = rst.collect()

    assert(GeometryUDT.FromWkt(collect(0).getAs[GeometryUDT](0).toString).getArea == 1.0)

    val data2 = Seq(
      Row(GeometryUDT.FromWkt("POINT (-5 -5)")),
      Row(GeometryUDT.FromWkt("LINESTRING (0 0, 10 10, 20 20)")),
    )

    val df2 = spark.createDataFrame(spark.sparkContext.parallelize(data2), schema)
    df2.createOrReplaceTempView("ST_Union_Aggr_data2")

    val rst2 = spark.sql("select ST_Union_Aggr(geo) from ST_Union_Aggr_data2")
    rst2.show(false)

    val collect2 = rst2.collect()

    assert(GeometryUDT.FromWkt(collect2(0).getAs[GeometryUDT](0).toString).getArea == 0.0)

    val data3 = Seq(
      Row(GeometryUDT.FromWkt("POLYGON ((1 1,2 1,2 2,1 2,1 1))")),
      Row(GeometryUDT.FromWkt("LINESTRING (0 0, 10 10, 20 20)")),
    )

    val df3 = spark.createDataFrame(spark.sparkContext.parallelize(data3), schema)
    df3.createOrReplaceTempView("ST_Union_Aggr_data3")

    val rst3 = spark.sql("select ST_Union_Aggr(geo) from ST_Union_Aggr_data3")
    rst3.show(false)

    val collect3 = rst3.collect()

    assert(GeometryUDT.FromWkt(collect3(0).getAs[GeometryUDT](0).toString).getArea==1.0)

    val data4 = Seq(
      Row(GeometryUDT.FromWkt("POINT (-5 -5)")),
      Row(GeometryUDT.FromWkt("POINT (-1 -1)")),
    )

    val df4 = spark.createDataFrame(spark.sparkContext.parallelize(data4), schema)
    df4.createOrReplaceTempView("ST_Union_Aggr_data4")

    val rst4 = spark.sql("select ST_Union_Aggr(geo) from ST_Union_Aggr_data4")
    rst4.show(false)

    val collect4 = rst4.collect()

    assert(GeometryUDT.FromWkt(collect4(0).getAs[GeometryUDT](0).toString).getArea==0.0)
  }

  test("ST_Union_Aggr-Null") {
    val data = Seq(
      Row("POLYGON EMPTY"),
      Row("POLYGON EMPTY"),
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

    assert(GeometryUDT.FromWkt(collect(0).getAs[GeometryUDT](0).toString).getArea == 2.0)

    val rst2 = df.agg(st_union_aggr(st_geomfromtext(col("geo"))))
    rst2.show(false)

    val collect2 = rst2.collect()

    assert(GeometryUDT.FromWkt(collect2(0).getAs[GeometryUDT](0).toString).getArea == 2.0)
  }

  test("ST_Union_Aggr-Perf") {
    val r = scala.util.Random
    val dataNum = 100
    val randomRange = 50

    var data = List[Row]()
    for (a <- 0 to dataNum) {
      data = data :+ Row(GeometryUDT.FromWkt(s"""POINT (${r.nextInt(randomRange)} ${r.nextInt(randomRange)})"""))
    }

    for (a <- 0 to dataNum) {
      val minX = r.nextInt(randomRange)
      val minY = r.nextInt(randomRange)
      data = data :+ Row(GeometryUDT.FromWkt(s"""LINESTRING (${minX} ${minY}, ${minX + 10} ${minY + 10}, ${minX + 20} ${minY + 20})"""))
    }

    for (a <- 0 to dataNum) {
      val minX = r.nextInt(randomRange)
      val minY = r.nextInt(randomRange)
      val maxX = minX + 2
      val maxY = minY + 2
      data = data :+ Row(GeometryUDT.FromWkt(s"""POLYGON (($minX $minY, $maxX $minY, $maxX $maxY, $minX $maxY, $minX $minY))"""))
    }

    val rdd_d = spark.sparkContext.parallelize(data)
    val schema = StructType(Array(StructField("geo", new GeometryUDT, nullable = false)))
    val df = spark.createDataFrame(rdd_d, schema)
    df.createOrReplaceTempView("Perf_data")
    spark.sql("cache table Perf_data")

    val t1 = System.currentTimeMillis

    val rst = spark.sql("select ST_Union_Aggr(geo) from Perf_data")
    rst.createOrReplaceTempView("res")
    spark.sql("cache table res")

    val t2 = System.currentTimeMillis

    rst.show(false)
    println((t2 - t1) / 1000.0 + " secs")
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

  test("ST_Envelope_Aggr-AllNull") {
    val data = Seq(
      Row("error geo"),
      Row("LINESTRING EMPTY"),
      Row(null),
      Row("POLYGON EMPTY"),
      Row("MULTIPOLYGON EMPTY"),
    )

    val schema = StructType(Array(StructField("geo", StringType, nullable = true)))
    val df = spark.createDataFrame(spark.sparkContext.parallelize(data), schema)
    df.createOrReplaceTempView("ST_Envelope_Aggr_Null_data")

    val rst = spark.sql("select ST_Envelope_Aggr(ST_GeomFromText(geo)) from ST_Envelope_Aggr_Null_data")
    rst.show(false)


    val collect = rst.collect()
    assert(collect(0).getAs[GeometryUDT](0).toString == "POLYGON EMPTY")

    val rst2 = df.agg(st_envelope_aggr(st_geomfromtext(col("geo"))))
    rst2.show(false)
    val collect2 = rst2.collect()

    assert(collect2(0).getAs[GeometryUDT](0).toString == "POLYGON EMPTY")

  }
}

