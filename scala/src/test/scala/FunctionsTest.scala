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
import org.apache.spark.sql.arctern.expressions.{ST_GeomFromText, ST_Within}

class FunctionsTest extends AdapterTest {
  test("ST_Within") {
    spark.sessionState.functionRegistry.createOrReplaceTempFunction("ST_GeomFromText", ST_GeomFromText)
    spark.sessionState.functionRegistry.createOrReplaceTempFunction("ST_Within", ST_Within)

    val data = Seq(
      Row(1, "POINT (20 20)", "POLYGON ((0 0, 40 0, 40 40, 0 40, 0 0))"),
      Row(2, "POINT (50 50)", "POLYGON ((0 0, 40 0, 40 40, 0 40, 0 0))"),
      Row(3, "POLYGON ((10 10, 20 10, 20 20, 10 20, 10 10))", "POLYGON ((0 0, 40 0, 40 40, 0 40, 0 0))"),
      Row(4, "POLYGON ((10 10, 50 10, 50 50, 10 50, 10 10))", "POLYGON ((0 0, 40 0, 40 40, 0 40, 0 0))")
    )

    val rdd_d = spark.sparkContext.parallelize(data)
    val schema = StructType(Array(StructField("idx", IntegerType, nullable = false), StructField("geo1", StringType, nullable = false), StructField("geo2", StringType, nullable = false)))
    val df = spark.createDataFrame(rdd_d, schema)
    df.createOrReplaceTempView("table_ST_Within")
    val rst = spark.sql("select idx, ST_Within(ST_GeomFromText(geo1), ST_GeomFromText(geo2)) from table_ST_Within")

    //    rst.queryExecution.debug.codegen()
    val collect = rst.collect()

    assert(collect(0).getBoolean(1) == true)
    assert(collect(1).getBoolean(1) == false)
    assert(collect(2).getBoolean(1) == true)
    assert(collect(3).getBoolean(1) == false)

    rst.show(false)
  }

  test("ST_Within With Null") {
    spark.sessionState.functionRegistry.createOrReplaceTempFunction("ST_GeomFromText", ST_GeomFromText)
    spark.sessionState.functionRegistry.createOrReplaceTempFunction("ST_Within", ST_Within)

    val data = Seq(
      Row(1, null, "POLYGON ((0 0, 40 0, 40 40, 0 40, 0 0))"),
      Row(2, "POINT (50 50)", null),
      Row(3, null, null),
      Row(4, "POLYGON ((10 10, 50 10, 50 50, 10 50, 10 10))", "POLYGON ((0 0, 40 0, 40 40, 0 40, 0 0))")
    )

    val rdd_d = spark.sparkContext.parallelize(data)
    val schema = StructType(Array(StructField("idx", IntegerType, nullable = false), StructField("geo1", StringType, nullable = true), StructField("geo2", StringType, nullable = true)))
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
}
