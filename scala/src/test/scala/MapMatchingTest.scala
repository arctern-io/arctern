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
import org.apache.spark.sql.arctern.{GeometryUDT, MapMatching}
import org.apache.spark.sql.arctern.index.RTreeIndex
import org.apache.spark.sql.types.{StructField, StructType}
import org.locationtech.jts.io.WKTReader

class MapMatchingTest extends AdapterTest {
  test("test index gdfgfhfgjfg") {
    val nr = new MapMatching
    nr.compute2()
  }
}
