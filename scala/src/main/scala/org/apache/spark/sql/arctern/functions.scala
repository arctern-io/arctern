package org.apache.spark.sql.arctern

import org.apache.spark.sql.Column

object functions {
  def st_point(x: Column, y: Column): Column = Column {
    expressions.ST_Point(Seq(x.expr, y.expr))
  }

}
