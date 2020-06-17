package org.apache.spark.sql.arctern.index

import java.util

import org.apache.spark.sql.arctern.GeometryUDT
import org.locationtech.jts.geom.{Envelope, Geometry}
import org.locationtech.jts.index.quadtree.Quadtree
import org.locationtech.jts.index.strtree.STRtree
import org.locationtech.jts.index.{ItemVisitor, SpatialIndex}

class IndexBuilder(indexType: String) extends SpatialIndex with Serializable {

  var index:SpatialIndex = null
  if (indexType.equalsIgnoreCase("RTREE"))
    index = new STRtree()
  else
    index = new Quadtree()

  override def insert(itemEnv: Envelope, item: Any): Unit = index.insert(itemEnv, item)

  def insert (array: Array[Geometry]): Unit = {
    array.foreach { geo =>
      val env = geo.getEnvelopeInternal
      index.insert(env, geo)
    }
  }

  override def query(searchEnv: Envelope): util.List[_] = index.query(searchEnv)

  override def query(searchEnv: Envelope, visitor: ItemVisitor): Unit = index.query(searchEnv, visitor)

  override def remove(itemEnv: Envelope, item: Any): Boolean = index.remove(itemEnv, item)
}

