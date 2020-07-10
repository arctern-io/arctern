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
import org.apache.spark.sql.arctern.functions._
import org.apache.spark.sql.arctern.{GeometryType, GeometryUDT}
import org.apache.spark.sql.types._

//case class Info(text: String)

class CornerTest extends AdapterTest {
  test("geom_as_source") {
    // TODO: currently buggy
    val ss = spark
    import ss.implicits._
    val data = Seq(
      Row(GeometryUDT.FromWkt("POINT (5 5)")),
      Row(GeometryUDT.FromWkt("POINT (1 1)")),
      Row(null),
    )
    val rdd = spark.sparkContext.parallelize(data)
    val schema = StructType(Seq(StructField("text1", GeometryType)))
    val df1 = spark.createDataFrame(rdd, schema)
    df1.createOrReplaceTempView("source")
    val df2 = df1.select(st_equals('text1, 'text1).as("c"))
    df2.show()
  }

  test("text_as_source_and_reuse_text") {
    val ss = spark
    import ss.implicits._
    val data = Seq(
      Row("POINT (5 5)"),
      Row("POINT (1 1)"),
      Row(null),
    )
    val rdd = spark.sparkContext.parallelize(data)
    val schema = StructType(Seq(StructField("text1", StringType)))
    val df1 = spark.createDataFrame(rdd, schema)
    df1.createOrReplaceTempView("source")
    val df2 = df1.select(st_equals(st_geomfromtext('text1), st_geomfromtext('text1)).as("c"))
    df2.show()
  }
  test("text_as_source_and_reuse_geom") {
    val ss = spark
    import ss.implicits._
    val data = Seq(
      Row("POINT (5 5)"),
      Row("POINT (1 1)"),
      Row(null),
    )
    val rdd = spark.sparkContext.parallelize(data)
    val schema = StructType(Seq(StructField("text", StringType)))
    val df1 = spark.createDataFrame(rdd, schema)
    df1.createOrReplaceTempView("source")

    val df2 = df1.select(st_geomfromtext('text).as("geom"))
    val df3 = df2.select(st_equals('geom, 'geom).as("c"))
    df3.show()
  }
}

