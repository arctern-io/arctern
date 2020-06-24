package org.apache.spark.sql.arctern.expressions

import org.apache.spark.sql.arctern.GeometryType
import org.apache.spark.sql.catalyst.dsl.expressions._
import org.apache.spark.sql.catalyst.expressions._
import org.apache.spark.sql.catalyst.expressions.aggregate.DeclarativeAggregate
import org.apache.spark.sql.catalyst.expressions.codegen.{CodegenContext, ExprCode}
import org.apache.spark.sql.types._


case class GeometryEnvelope(expression: Expression) extends ST_UnaryOp {

  override def expr: Expression = expression

  override def nullable: Boolean = false

  override protected def doGenCode(ctx: CodegenContext, ev: ExprCode): ExprCode = codeGenJob(ctx, ev, geo => s"org.apache.spark.sql.arctern.expressions.utils.envelopeAsList($geo)")

  override def dataType: DataType = ArrayType(DoubleType)

  override def inputTypes: Seq[AbstractDataType] = Seq(GeometryType)
}

case class GetElementByIndex(index: Int, elementDataType: DataType, input: Expression) extends UnaryExpression with ExpectsInputTypes {
  private val arrayType = ArrayType(elementDataType, false)

  override def inputTypes: Seq[AbstractDataType] = Seq(arrayType)

  override def dataType: DataType = elementDataType

  override def child: Expression = input

  override def toString(): String = s"$child[$index]"

  override def nullSafeEval(input: Any): Any = throw new Exception("no eval")

  override def doGenCode(ctx: CodegenContext, ev: ExprCode): ExprCode = {
    defineCodeGen(ctx, ev, eval => s"${eval}.getDouble($index)")
  }
}

case class dslMin(leftExpr: Expression, rightExpr: Expression) extends BinaryExpression with ExpectsInputTypes {
  override def inputTypes: Seq[AbstractDataType] = Seq(DoubleType, DoubleType)

  override def dataType: DataType = DoubleType

  override def left: Expression = leftExpr

  override def right: Expression = rightExpr

  override def toString(): String = s"min($left, $right)"

  override def nullSafeEval(left: Any, right: Any): Any = throw new Exception("no eval")

  override def doGenCode(ctx: CodegenContext, ev: ExprCode): ExprCode = {
    defineCodeGen(ctx, ev, (eval1, eval2) => s"$eval1 < $eval2 ? $eval1 : $eval2")
  }
}

case class dslMax(leftExpr: Expression, rightExpr: Expression) extends BinaryExpression with ExpectsInputTypes {
  override def inputTypes: Seq[AbstractDataType] = Seq(DoubleType, DoubleType)

  override def dataType: DataType = DoubleType

  override def left: Expression = leftExpr

  override def right: Expression = rightExpr

  override def toString(): String = s"max($left, $right)"

  override def nullSafeEval(left: Any, right: Any): Any = throw new Exception("no eval")

  override def doGenCode(ctx: CodegenContext, ev: ExprCode): ExprCode = {
    defineCodeGen(ctx, ev, (eval1, eval2) => s"$eval1 > $eval2 ? $eval1 : $eval2")
  }
}

object EnvelopeAggr {
  val emptyResponse = ST_GeomFromText(Seq(Literal("Polygon Empty")))
}

case class EnvelopeAggr(geom: Expression)
  extends DeclarativeAggregate with ImplicitCastInputTypes {
  override def children: Seq[Expression] = Seq(geom)

  override def nullable: Boolean = false

  override def dataType: DataType = GeometryType

  override def inputTypes: Seq[AbstractDataType] = Seq(GeometryType)

  protected val minX = AttributeReference("minX", DoubleType, nullable = false)()
  protected val minY = AttributeReference("minY", DoubleType, nullable = false)()
  protected val maxX = AttributeReference("maxX", DoubleType, nullable = false)()
  protected val maxY = AttributeReference("maxY", DoubleType, nullable = false)()
  protected val envelope = Seq(minX, minY, maxX, maxY)
  override val aggBufferAttributes: Seq[AttributeReference] = envelope

  val negInf = Literal(scala.Double.NegativeInfinity)
  val posInf = Literal(scala.Double.PositiveInfinity)
  override val initialValues: Seq[Expression] = {
    Seq(posInf, posInf, negInf, negInf)
  }

  override lazy val updateExpressions: Seq[Expression] = updateExpressionDef

  //  def dslMin(e1: Expression, e2: Expression): Expression = If(e1 < e2, e1, e2)
  //
  //  def dslMax(e1: Expression, e2: Expression): Expression = If(e1 > e2, e1, e2)

  override val mergeExpressions: Seq[Expression] = {
    def getMin(attr: AttributeReference): Expression = dslMin(attr.left, attr.right)

    def getMax(attr: AttributeReference): Expression = dslMax(attr.left, attr.right)

    Seq(getMin(minX), getMin(minY), getMax(maxX), getMax(maxY))
  }

  protected def updateExpressionDef: Seq[Expression] = {
    val input_envelope = GeometryEnvelope(geom)
    val input_minX = GetElementByIndex(0, DoubleType, input_envelope)
    val input_minY = GetElementByIndex(1, DoubleType, input_envelope)
    val input_maxX = GetElementByIndex(2, DoubleType, input_envelope)
    val input_maxY = GetElementByIndex(3, DoubleType, input_envelope)
    Seq(
      dslMin(minX, input_minX),
      dslMin(minY, input_minY),
      dslMax(maxX, input_maxX),
      dslMax(maxY, input_maxY),
    )
  }

  override val evaluateExpression: Expression = {
    val condition = envelope(0) <= envelope(2)
    val data = ST_PolygonFromEnvelope(envelope)
    If(condition, data, EnvelopeAggr.emptyResponse)
  }

  override def prettyName: String = "ST_Envelope_Aggr"
}


