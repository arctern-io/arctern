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
import org.apache.spark.sql.arctern.expressions.ST_GeomFromText

class ConstructorsTest extends AdapterTest {
  test("ST_GeomFromText With Null") {

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

}
