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
package org.apache.spark.sql.arctern.expressions

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.sql.arctern.index.IndexBuilder
import org.locationtech.jts.geom.Geometry

import scala.util.control.Breaks._

case class IndexedJoin(broadcast: Broadcast[IndexBuilder]) {
  def index = broadcast.value

  def join(input: Array[Geometry]): Array[Geometry] = {
    var result = new Array[Geometry](input.size)
    input.indices.foreach { i =>
      val geo_search = input(i)
      val env = geo_search.getEnvelopeInternal
      val geo_list = index.query(env)
      breakable {
        geo_list.forEach { geo =>
          val loop_geo = geo.asInstanceOf[Geometry]
          if (geo_search.intersects(loop_geo)) {
            //TODO::mutil indexed result intersects with searched geometry
            result(i) = loop_geo
            break
          }
        }
      }
    }
    result
  }
}
