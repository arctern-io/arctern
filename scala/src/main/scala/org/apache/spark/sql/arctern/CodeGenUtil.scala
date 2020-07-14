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
package org.apache.spark.sql.arctern

import org.apache.spark.sql.catalyst.expressions.codegen.{CodegenContext, ExprCode}
import org.apache.spark.sql.catalyst.expressions.{ExpectsInputTypes, Expression}
import org.apache.spark.sql.types.DataType

abstract class ArcternExpr extends Expression with ExpectsInputTypes {
  def isArcternExpr = true
}

object CodeGenUtil {
  private def geometryFromArcternExpr(codeString: String) = {
    val serialKeyWords = s"${GeometryUDT.getClass().getName().dropRight(1)}.GeomSerialize"

    val serialIdx = codeString.lastIndexOf(serialKeyWords)
    if (serialIdx == -1) {
      throw new RuntimeException(s"can't find $serialKeyWords in the code string")
    }

    val leftBracketIdx = codeString.indexOf("(", serialIdx + serialKeyWords.length())
    if (leftBracketIdx == -1) {
      throw new RuntimeException("leftBracketIdx = -1")
    }

    val rightBracketIdx = codeString.indexOf(")", leftBracketIdx)
    if (rightBracketIdx == -1) {
      throw new RuntimeException("rightBracketIdx = -1")
    }

    val geoName = codeString.slice(leftBracketIdx + 1, rightBracketIdx).trim()

    var serialStatementIdx = serialIdx
    while (serialStatementIdx > 0 && codeString(serialStatementIdx) != ';') serialStatementIdx = serialStatementIdx - 1
    if (serialStatementIdx < 0) {
      throw new RuntimeException(s"serialStatementIdx = $serialStatementIdx")
    }

    var serialStatementEnd = rightBracketIdx
    while (serialStatementEnd < codeString.length() && codeString(serialStatementEnd) != ';') serialStatementEnd = serialStatementEnd + 1
    if (serialStatementEnd >= codeString.length()) {
      throw new RuntimeException(s"serialStatementEnd = $serialStatementEnd")
    }

    val geoDeclare = mutableGeometryInitCode(geoName)
    val geoDeclateIdx = codeString.indexOf(geoDeclare)

    val newCodeString = codeString.slice(0, geoDeclateIdx) +
      codeString.slice(geoDeclateIdx + geoDeclare.length(), serialStatementIdx + 1) +
      codeString.slice(serialStatementEnd + 1, codeString.length)

    (geoName, geoDeclare, newCodeString)
  }

  private def geometryFromNormalExpr(exrCode: ExprCode, geoName: String) = {
    val geoDeclare = mutableGeometryInitCode(geoName)
    val newCodeString =
      s"""
         |${exrCode.code}
         |$geoName = ${deserializeGeometryCode(exrCode.value)}
                         """.stripMargin
    (geoName, geoDeclare, newCodeString)
  }

  def geometryFromExpr(ctx: CodegenContext, expr: Expression, exprCode: ExprCode): (String, String, String) = {
    if (CodeGenUtil.isArcternExpr(expr)) CodeGenUtil.geometryFromArcternExpr(exprCode.code.toString())
    else CodeGenUtil.geometryFromNormalExpr(exprCode, ctx.freshName(exprCode.value))
  }

  def assignmentCode(callFunc: String, value: String, geoName: String, dt: DataType) = {
    dt match {
      case _: GeometryUDT =>
        s"""
           |${mutableGeometryInitCode(geoName)}
           |${geoName} = $callFunc;
           |${emptyPointCheck(geoName)}
           |$value = ${serialGeometryCode(geoName)}
           |""".stripMargin
      case _ => s"$value = $callFunc;"
    }
  }

  def mutableGeometryInitCode(geo_name: String) = s"org.locationtech.jts.geom.Geometry $geo_name = null;"

  def serialGeometryCode(geo_code: String) = s"${GeometryUDT.getClass().getName().dropRight(1)}.GeomSerialize($geo_code);"

  def deserializeGeometryCode(geo_code: String) = s"${GeometryUDT.getClass().getName().dropRight(1)}.GeomDeserialize($geo_code);"

  def isGeometryExpr(expr: Expression): Boolean = expr.dataType match {
    case _: GeometryUDT => true
    case _ => false
  }

  def isArcternExpr(expr: Expression): Boolean = expr match {
    case _: ArcternExpr => true
    case _ => false
  }

  def utf8StringFromStringCode(string_name: String) = s"org.apache.spark.unsafe.types.UTF8String.fromString($string_name);"

  def emptyPointCheck(geoName: String) = s"""if ($geoName.getGeometryType() == "Point" && $geoName.isEmpty()) $geoName = new org.locationtech.jts.geom.GeometryFactory().createGeometryCollection();"""
}
