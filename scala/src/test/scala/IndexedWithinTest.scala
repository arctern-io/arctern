import org.apache.spark.sql.arctern.functions._
import org.apache.spark.sql.arctern.index.RTreeIndex
import org.apache.spark.sql.functions._
import org.locationtech.jts.geom.{Envelope, Geometry}

import scala.collection.mutable


case class Info(text: String)

class IndexedWithinTest extends AdapterTest {
  test("naive") {
    val ss = spark
    import ss.implicits._
    val points_text = Seq(
      Info("Point(1 1)"),
      Info("Point(1 2)"),
      Info("Point(2 1)"),
      Info("Point(2 2)"),
      Info("Point(3 3)"),
      Info("Point(4 5)"),
      Info("Point(8 8)"),
      Info("Point(10 10)"),
    ).toDF.withColumn("attr", monotonically_increasing_id())

    val polygons_text = Seq(
      Info("Polygon((0 0, 3 0, 3.1 3.1, 0 3, 0 0))"),
      Info("Polygon((6 6, 3 6, 2.9 2.9, 6 3, 6 6))"),
      Info("Polygon((6 6, 9 6, 9 9, 6 9, 6 6))"),
    ).toDF.withColumn("id", monotonically_increasing_id())

    val points = points_text.select('attr * 100, st_geomfromtext('text).as("points"))

    val polygons = polygons_text.select('id, st_geomfromtext('text).as("polygons"))

    // here feed dog

    val index_data = polygons.select('id, st_envelopinternal('polygons).as("envs"))


    val tree = new RTreeIndex


    index_data.collect().foreach {
      row => {
        val id = row.getAs[Long](0)
        val data = row.getAs[mutable.WrappedArray[Double]](1)
        val env = new Envelope(data(0), data(2), data(1), data(3))
        tree.insert(env, id)
      }
    }

    val broadcast = spark.sparkContext.broadcast(tree)

    val points_rdd = points.as[(Long, Geometry)].rdd
    val polygon_rdd = polygons.as[(Long, Geometry)].rdd

    val cross = points_rdd.flatMap {
      tp => {
        val (attr, point) = tp
        val env = point.getEnvelopeInternal
        val polyList = broadcast.value.query(env)
        polyList.toArray.map(poly => (poly.asInstanceOf[Long], (attr, point)))
      }
    }

//    print(cross.collect().toSeq)
//    print(polygon_rdd.collect().toSeq)
    val fin = cross.join(polygon_rdd).flatMap{
      tp => {
        val polygonId = tp._1
        val ((pointId, point), polygon) = tp._2
        if (point.within(polygon)) {
          (pointId, polygonId)::Nil
        } else {
          Nil
        }
      }
    }

    print(fin.collect().toSeq)
    // both are unusable
  }
}
