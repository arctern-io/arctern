import org.apache.spark.sql.arctern.MapMatching
import org.apache.spark.sql.arctern.index.RTreeIndex
import org.locationtech.jts.io.WKTReader

class MapMatchingTest extends AdapterTest {
  test("test index gdfgfhfgjfg") {
    val nr = new MapMatching
    nr.compute()
  }
}
