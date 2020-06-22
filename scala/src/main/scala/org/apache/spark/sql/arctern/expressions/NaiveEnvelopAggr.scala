package org.apache.spark.sql.arctern.expressions

import org.apache.spark.sql.catalyst.dsl.expressions._
import org.apache.spark.sql.catalyst.expressions._
import org.apache.spark.sql.catalyst.expressions.aggregate.DeclarativeAggregate
import org.apache.spark.sql.types._

case class NaiveEnvelopAggr(geom: Expression)
  extends DeclarativeAggregate with ImplicitCastInputTypes {
  override def children: Seq[Expression] = Seq(geom)

  override def nullable: Boolean = true

  override def dataType: DataType = DoubleType // TODO: use GeometryUDT
  override def inputTypes: Seq[AbstractDataType] = Seq(DoubleType) // TODO: use GeometryUDT

  protected val minX = AttributeReference("minX", DoubleType, nullable = false)()
  protected val minY = AttributeReference("minY", DoubleType, nullable = false)()
  protected val maxX = AttributeReference("maxX", DoubleType, nullable = false)()
  protected val maxY = AttributeReference("maxY", DoubleType, nullable = false)()
  protected val envelop = Seq(minX, minY, maxX, maxY)
  override val aggBufferAttributes: Seq[AttributeReference] = envelop
  override val initialValues: Seq[Expression] = {
    val negInf = Literal(scala.Double.NegativeInfinity)
    val posInf = Literal(scala.Double.PositiveInfinity)
    Seq(posInf, posInf, negInf, negInf)
  }

  override lazy val updateExpressions: Seq[Expression] = updateExpressionDef

  def dslMin(e1: Expression, e2: Expression): Expression = If(e1 < e2, e1, e2)
  def dslMax(e1: Expression, e2: Expression): Expression = If(e1 > e2, e1, e2)

  override val mergeExpressions: Seq[Expression] = {
    def getMin(attr: AttributeReference): Expression = dslMin(attr.left, attr.right)

    def getMax(attr: AttributeReference): Expression = dslMax(attr.left, attr.right)

    Seq(getMin(minX), getMin(minY), getMax(maxX), getMax(maxY))
  }

  protected def updateExpressionDef: Seq[Expression] = {
    // TODO: use GeometryUDT and jtx, currently a fake one
    val input_envelop: Seq[Expression] = Seq(geom, Literal(0.0), geom + 5, Literal(1.0))
    Seq(
      dslMin(envelop(0), input_envelop(0)),
      dslMin(envelop(1), input_envelop(1)),
      dslMax(envelop(2), input_envelop(2)),
      dslMax(envelop(3), input_envelop(3)),
    )
  }

  override val evaluateExpression: Expression = {
    // TODO: convert back to envelop
    (maxX - minX) * (maxY - minY)
  }

  override def prettyName: String = "naive_envelop_aggr"
}


