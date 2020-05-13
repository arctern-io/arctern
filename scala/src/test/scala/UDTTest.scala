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

import org.scalatest.FunSuite
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.log4j.Logger
import org.apache.log4j.Level
import org.apache.spark.sql.types._
import org.apache.spark.sql.arctern._
import org.locationtech.jts.io.WKTReader

class UDTTest extends FunSuite {
  test("GeometryUDT") {

    Logger.getLogger("org").setLevel(Level.WARN)
    val spark = SparkSession.builder()
      .master("local[*]")
      .appName("geometry_udt_test")
      .getOrCreate()

    UdtRegistratorWrapper.registerUDT()

    val data = Seq(
      Row(1, new WKTReader().read("POINT (10 20)")),
      Row(2, new WKTReader().read("LINESTRING (0 0, 10 10, 20 20)")),
      Row(3, new WKTReader().read("POLYGON ((30 10, 40 40, 20 40, 10 20, 30 10))")),
      Row(4, new WKTReader().read("MULTIPOINT ((10 40), (40 30), (20 20), (30 10))")),
      Row(5, new WKTReader().read("MULTIPOLYGON (((30 20, 45 40, 10 40, 30 20)), ((15 5, 40 10, 10 20, 5 10, 15 5)))"))
    )

    val rdd_d = spark.sparkContext.parallelize(data)
    val schema = StructType(Array(StructField("idx", IntegerType, nullable = false), StructField("geometry", new GeometryUDT, nullable = false)))
    val df = spark.createDataFrame(rdd_d, schema)
    df.createOrReplaceTempView("data")
    val rst = spark.sql("select * from data")
    rst.show(false)

    spark.stop()
  }
}
