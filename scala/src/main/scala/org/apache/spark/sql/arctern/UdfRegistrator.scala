package org.apache.spark.sql.arctern

import org.apache.spark.sql.{AnalysisException, SparkSession}
import org.apache.spark.sql.arctern.expressions._
import org.apache.spark.sql.catalyst.analysis.FunctionRegistry.{FunctionBuilder}
import org.apache.spark.sql.catalyst.expressions.{Expression, ExpressionDescription, ExpressionInfo, RuntimeReplaceable}

import scala.reflect.ClassTag

object UdfRegistrator {
  def register(spark: SparkSession) = {
    // Register constructor UDFs
    spark.sessionState.functionRegistry.createOrReplaceTempFunction("ST_GeomFromText", ST_GeomFromText)
    spark.sessionState.functionRegistry.createOrReplaceTempFunction("ST_Point", ST_Point)
    spark.sessionState.functionRegistry.createOrReplaceTempFunction("ST_PolygonFromEnvelope", ST_PolygonFromEnvelope)
    spark.sessionState.functionRegistry.createOrReplaceTempFunction("ST_GeomFromGeoJSON", ST_GeomFromGeoJSON)
    spark.sessionState.functionRegistry.createOrReplaceTempFunction("ST_AsText", ST_AsText)
    spark.sessionState.functionRegistry.createOrReplaceTempFunction("ST_AsGeoJSON", ST_AsGeoJSON)
    // Register function UDFs
    spark.sessionState.functionRegistry.createOrReplaceTempFunction("ST_Within", ST_Within)
    spark.sessionState.functionRegistry.createOrReplaceTempFunction("ST_Centroid", ST_Centroid)
    spark.sessionState.functionRegistry.createOrReplaceTempFunction("ST_IsValid", ST_IsValid)
    spark.sessionState.functionRegistry.createOrReplaceTempFunction("ST_GeometryType", ST_GeometryType)
    spark.sessionState.functionRegistry.createOrReplaceTempFunction("ST_IsSimple", ST_IsSimple)
    spark.sessionState.functionRegistry.createOrReplaceTempFunction("ST_NPoints", ST_NPoints)
    spark.sessionState.functionRegistry.createOrReplaceTempFunction("ST_Envelope", ST_Envelope)
    spark.sessionState.functionRegistry.createOrReplaceTempFunction("ST_Buffer", ST_Buffer)
    spark.sessionState.functionRegistry.createOrReplaceTempFunction("ST_PrecisionReduce", ST_PrecisionReduce)
    spark.sessionState.functionRegistry.createOrReplaceTempFunction("ST_Intersection", ST_Intersection)
    spark.sessionState.functionRegistry.createOrReplaceTempFunction("ST_SimplifyPreserveTopology", ST_SimplifyPreserveTopology)
    spark.sessionState.functionRegistry.createOrReplaceTempFunction("ST_ConvexHull", ST_ConvexHull)
    spark.sessionState.functionRegistry.createOrReplaceTempFunction("ST_Area", ST_Area)
    spark.sessionState.functionRegistry.createOrReplaceTempFunction("ST_Length", ST_Length)
    spark.sessionState.functionRegistry.createOrReplaceTempFunction("ST_HausdorffDistance", ST_HausdorffDistance)
    spark.sessionState.functionRegistry.createOrReplaceTempFunction("ST_Distance", ST_Distance)
    spark.sessionState.functionRegistry.createOrReplaceTempFunction("ST_Equals", ST_Equals)
    spark.sessionState.functionRegistry.createOrReplaceTempFunction("ST_Touches", ST_Touches)
    spark.sessionState.functionRegistry.createOrReplaceTempFunction("ST_Overlaps", ST_Overlaps)
    spark.sessionState.functionRegistry.createOrReplaceTempFunction("ST_Crosses", ST_Crosses)
    spark.sessionState.functionRegistry.createOrReplaceTempFunction("ST_Contains", ST_Contains)
    spark.sessionState.functionRegistry.createOrReplaceTempFunction("ST_Intersects", ST_Intersects)
    spark.sessionState.functionRegistry.createOrReplaceTempFunction("ST_DistanceSphere", ST_DistanceSphere)
    spark.sessionState.functionRegistry.createOrReplaceTempFunction("ST_Transform", ST_Transform)
    spark.sessionState.functionRegistry.createOrReplaceTempFunction("ST_MakeValid", ST_MakeValid)
    spark.sessionState.functionRegistry.createOrReplaceTempFunction("ST_CurveToLine", ST_CurveToLine)

    // TODO: remove ExpressionInfo
    //    val (naive_name, (naive_info, naive_builder)) = expression[NaiveEnvelopeAggr]("naive_envelope_aggr")
    val naive_builder = (seq: Seq[Expression]) =>  {
      assert(seq.size == 1)
      NaiveEnvelopeAggr(seq(0))
    }
    spark.sessionState.functionRegistry.createOrReplaceTempFunction("naive_envelope_aggr", naive_builder)

    // Register aggregate function UDFs
    spark.udf.register("ST_Union_Aggr", new ST_Union_Aggr)
    spark.udf.register("ST_Envelope_Aggr", new ST_Envelope_Aggr)
  }


//   // TODO: copy from spark, need to simplify
//   private def expression[T <: Expression](name: String)
//                                          (implicit tag: ClassTag[T]): (String, (ExpressionInfo, FunctionBuilder)) = {

//     // For `RuntimeReplaceable`, skip the constructor with most arguments, which is the main
//     // constructor and contains non-parameter `child` and should not be used as function builder.
//     val constructors = if (classOf[RuntimeReplaceable].isAssignableFrom(tag.runtimeClass)) {
//       val all = tag.runtimeClass.getConstructors
//       val maxNumArgs = all.map(_.getParameterCount).max
//       all.filterNot(_.getParameterCount == maxNumArgs)
//     } else {
//       tag.runtimeClass.getConstructors
//     }
//     // See if we can find a constructor that accepts Seq[Expression]
//     val varargCtor = constructors.find(_.getParameterTypes.toSeq == Seq(classOf[Seq[_]]))
//     val builder = (expressions: Seq[Expression]) => {
//       if (varargCtor.isDefined) {
//         // If there is an apply method that accepts Seq[Expression], use that one.
//         try {
//           val exp = varargCtor.get.newInstance(expressions).asInstanceOf[Expression]
//           exp
//         } catch {
//           // the exception is an invocation exception. To get a meaningful message, we need the
//           // cause.
//           case e: Exception => throw new AnalysisException(e.getCause.getMessage)
//         }
//       } else {
//         // Otherwise, find a constructor method that matches the number of arguments, and use that.
//         val params = Seq.fill(expressions.size)(classOf[Expression])
//         val f = constructors.find(_.getParameterTypes.toSeq == params).getOrElse {
//           val validParametersCount = constructors
//             .filter(_.getParameterTypes.forall(_ == classOf[Expression]))
//             .map(_.getParameterCount).distinct.sorted
//           val invalidArgumentsMsg = if (validParametersCount.length == 0) {
//             s"Invalid arguments for function $name"
//           } else {
//             val expectedNumberOfParameters = if (validParametersCount.length == 1) {
//               validParametersCount.head.toString
//             } else {
//               validParametersCount.init.mkString("one of ", ", ", " and ") +
//                 validParametersCount.last
//             }
//             s"Invalid number of arguments for function $name. " +
//               s"Expected: $expectedNumberOfParameters; Found: ${params.length}"
//           }
//           throw new AnalysisException(invalidArgumentsMsg)
//         }
//         try {
//           val exp = f.newInstance(expressions : _*).asInstanceOf[Expression]
//           exp
//         } catch {
//           // the exception is an invocation exception. To get a meaningful message, we need the
//           // cause.
//           case e: Exception => throw new AnalysisException(e.getCause.getMessage)
//         }
//       }
//     }

//     (name, (expressionInfo[T](name), builder))

//   }

//   // TODO: copy from spark, need to simplify
//   private def expressionInfo[T <: Expression : ClassTag](name: String): ExpressionInfo = {
//     val clazz = scala.reflect.classTag[T].runtimeClass
//     val df = clazz.getAnnotation(classOf[ExpressionDescription])
//     if (df != null) {
//       if (df.extended().isEmpty) {
//         new ExpressionInfo(
//           clazz.getCanonicalName,
//           null,
//           name,
//           df.usage(),
//           df.arguments(),
//           df.examples(),
//           df.note(),
// //          df.group(),  // TODO: uncomment me for spark 3.0
//           df.since(),
//           df.deprecated())
//       } else {
//         // This exists for the backward compatibility with old `ExpressionDescription`s defining
//         // the extended description in `extended()`.
//         new ExpressionInfo(clazz.getCanonicalName, null, name, df.usage(), df.extended())
//       }
//     } else {
//       new ExpressionInfo(clazz.getCanonicalName, name)
//     }
//   }
}
