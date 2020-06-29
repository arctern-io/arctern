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
