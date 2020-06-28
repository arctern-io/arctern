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
import org.apache.spark.sql.arctern._
import org.apache.spark.sql.types._
import org.locationtech.jts.io.WKTReader

class UDTTest extends AdapterTest {
  test("GeometryUDT") {
    val data = Seq(
      Row(1, new WKTReader().read("POINT (10 20)")),
      Row(2, new WKTReader().read("LINESTRING (0 0, 10 10, 20 20)")),
      Row(3, new WKTReader().read("POLYGON ((30 10, 40 40, 20 40, 10 20, 30 10))")),
      Row(4, new WKTReader().read("MULTIPOINT ((10 40), (40 30), (20 20), (30 10))")),
      Row(5, new WKTReader().read("MULTIPOLYGON (((30 20, 45 40, 10 40, 30 20)), ((15 5, 40 10, 10 20, 5 10, 15 5)))")),
      Row(6, null)
    )

    val rdd_d = spark.sparkContext.parallelize(data)
    val schema = StructType(Array(StructField("idx", IntegerType, nullable = false), StructField("geometry", new GeometryUDT, nullable = true)))
    val df = spark.createDataFrame(rdd_d, schema)
    df.createOrReplaceTempView("table_GeometryUDT")

    val rst = spark.sql("select * from table_GeometryUDT")
    rst.show(false)

    val collect = rst.collect()

    assert(collect(0).getAs[GeometryUDT](1).toString == "POINT (10 20)")
    assert(collect(1).getAs[GeometryUDT](1).toString == "LINESTRING (0 0, 10 10, 20 20)")
    assert(collect(2).getAs[GeometryUDT](1).toString == "POLYGON ((30 10, 40 40, 20 40, 10 20, 30 10))")
    assert(collect(3).getAs[GeometryUDT](1).toString == "MULTIPOINT ((10 40), (40 30), (20 20), (30 10))")
    assert(collect(4).getAs[GeometryUDT](1).toString == "MULTIPOLYGON (((30 20, 45 40, 10 40, 30 20)), ((15 5, 40 10, 10 20, 5 10, 15 5)))")
    assert(collect(5).isNullAt(1))
  }

  test("GeomtryUDT_null") {
    val wkb_null: Array[Byte] = Array(0, 1, 2)
    val wkt_null: String = "abcdef"
    assert(GeometryUDT.FromWkb(wkb_null) == null)
    assert(GeometryUDT.FromWkt(wkt_null) == null)
  }
}
