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
    //  val rst = spark.sql("select idx, geo1, geo2 from data")
  
    //  rst.queryExecution.debug.codegen()
     val collect = rst.collect()
  
     assert(collect(0).getBoolean(1) == true)
     assert(collect(1).getBoolean(1) == false)
     assert(collect(2).getBoolean(1) == true)
     assert(collect(3).getBoolean(1) == false)
  
     rst.show(false)
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
    rst.show()

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
    //    val rst = spark.sql("select idx, ST_GeomFromText(geo) from data")
    //    rst.queryExecution.debug.codegen()
    val collect = rst.collect()
    assert(collect(0).isNullAt(1))
    assert(collect(2).isNullAt(1))

  }

  test("ST_Within-Nest"){
    val data = Seq(
      Row(1, "polygon((0 0, 0 1,1 1, 1 0, 0 0))", "polygon((0 0, 0 1,1 1, 1 0, 0 0))"),
      Row(2, "error geo", "polygon((0 0, 0 1,1 1, 1 0, 0 0))"),
      Row(3, "polygon((0 0, 0 1,1 1, 1 0, 0 0))", "error geo")
    )
    val schema = StructType(Array(StructField("idx", IntegerType, false), StructField("geo1", StringType, false), StructField("geo2", StringType, false)))
    val df = spark.createDataFrame(spark.sparkContext.parallelize(data), schema)
    df.createOrReplaceTempView("data")
    val rst = spark.sql("select idx, st_within(st_centroid(ST_GeomFromText(geo1)), st_centroid(ST_GeomFromText(geo2))) from data")
    rst.queryExecution.debug.codegen()
    val collect = rst.collect()
    assert(collect(0).getBoolean(1))
    assert(collect(1).isNullAt(1))
    assert(collect(2).isNullAt(1))
    // rst.show()
  }

}
