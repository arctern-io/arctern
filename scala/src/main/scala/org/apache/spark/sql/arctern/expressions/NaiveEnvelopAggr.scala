package org.apache.spark.sql.arctern.expressions

import org.apache.spark.sql.arctern.GeometryType
import org.apache.spark.sql.catalyst.dsl.expressions._
import org.apache.spark.sql.catalyst.expressions._
import org.apache.spark.sql.catalyst.expressions.aggregate.DeclarativeAggregate
import org.apache.spark.sql.types._

case class NaiveEnvelopAggr(geom: Expression)
  extends DeclarativeAggregate with ImplicitCastInputTypes {
  override def children: Seq[Expression] = Seq(geom)

  override def nullable: Boolean = true

  override def dataType: DataType = GeometryType
  override def inputTypes: Seq[AbstractDataType] = Seq(GeometryType) // TODO: use GeometryUDT

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
    val input_envelope = ST_Envelope(Seq(geom))
    val input_minX = EnvelopeGet("MinX", input_envelope)
//    val input_minY = EnvelopeGet("MinY", input_envelope)
//    val input_maxX = EnvelopeGet("MaxX", input_envelope)
//    val input_maxY = EnvelopeGet("MaxY", input_envelope)
//    Seq(
//      dslMin(minX, input_minX),
//      dslMin(minY, input_minY),
//      dslMax(maxX, input_maxX),
//      dslMax(maxY, input_maxY),
//    )
    val newMinX = dslMin(minX, input_minX)
    Seq(newMinX, minY, maxX, maxY)
  }

  override val evaluateExpression: Expression = {
    ST_PolygonFromEnvelope(envelop)
  }

  override def prettyName: String = "naive_envelop_aggr"
}


