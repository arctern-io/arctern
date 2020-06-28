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
package org.apache.spark.sql.arctern.expressions

import org.apache.spark.sql.Row
import org.apache.spark.sql.arctern.GeometryUDT
import org.apache.spark.sql.expressions.{MutableAggregationBuffer, UserDefinedAggregateFunction}
import org.apache.spark.sql.types.{DataType, StructField, StructType}
import org.locationtech.jts.geom.{Coordinate, Geometry, GeometryFactory}

class ST_Union_Aggr extends UserDefinedAggregateFunction {
  override def inputSchema: StructType = StructType(StructField("Union", new GeometryUDT) :: Nil)

  override def bufferSchema: StructType = StructType(StructField("Union", new GeometryUDT) :: Nil)

  override def dataType: DataType = new GeometryUDT

  override def deterministic: Boolean = true

  override def initialize(buffer: MutableAggregationBuffer): Unit = {
    val coordinate = new Coordinate(-999999999, -999999999)
    buffer(0) = new GeometryFactory().createPoint(coordinate)
  }

  override def update(buffer: MutableAggregationBuffer, input: Row): Unit = {
    if (input.isNullAt(0)) return
    val accumulateUnion = buffer.getAs[Geometry](0)
    val newGeo = input.getAs[Geometry](0)
    if (accumulateUnion.getCoordinates()(0).x == -999999999) buffer(0) = newGeo
    else buffer(0) = accumulateUnion.union(newGeo)
  }

  override def merge(buffer1: MutableAggregationBuffer, buffer2: Row): Unit = {
    if (buffer2.isNullAt(0)) return
    val leftGeo = buffer1.getAs[Geometry](0)
    val rightGeo = buffer2.getAs[Geometry](0)
    if (leftGeo.getCoordinates()(0).x == -999999999) buffer1(0) = rightGeo
    else if (rightGeo.getCoordinates()(0).x == -999999999) buffer1(0) = leftGeo
    else buffer1(0) = leftGeo.union(rightGeo)
  }

  override def evaluate(buffer: Row): Any = buffer.getAs[Geometry](0)
}

////deprecated
//class ST_Envelope_Aggr extends UserDefinedAggregateFunction {
//  override def inputSchema: StructType = StructType(StructField("Envelope", new GeometryUDT) :: Nil)
//
//  override def bufferSchema: StructType = StructType(StructField("Envelope", new GeometryUDT) :: Nil)
//
//  override def dataType: DataType = new GeometryUDT
//
//  override def deterministic: Boolean = true
//
//  override def initialize(buffer: MutableAggregationBuffer): Unit = {
//    val coordinate = new Coordinate(-999999999, -999999999)
//    buffer(0) = new GeometryFactory().createPoint(coordinate)
//  }
//
//  override def update(buffer: MutableAggregationBuffer, input: Row): Unit = {
//    if (input.isNullAt(0)) return
//    val accumulateEnvelope = buffer.getAs[Geometry](0).getEnvelopeInternal
//    val newEnvelope = input.getAs[Geometry](0).getEnvelopeInternal
//    val coordinates: Array[Coordinate] = new Array[Coordinate](5)
//    var minX = 0.0
//    var minY = 0.0
//    var maxX = 0.0
//    var maxY = 0.0
//    if (accumulateEnvelope.getMinX == -999999999) {
//      minX = newEnvelope.getMinX
//      minY = newEnvelope.getMinY
//      maxX = newEnvelope.getMaxX
//      maxY = newEnvelope.getMaxY
//    } else {
//      minX = Math.min(accumulateEnvelope.getMinX, newEnvelope.getMinX)
//      minY = Math.min(accumulateEnvelope.getMinY, newEnvelope.getMinY)
//      maxX = Math.max(accumulateEnvelope.getMaxX, newEnvelope.getMaxX)
//      maxY = Math.max(accumulateEnvelope.getMaxY, newEnvelope.getMaxY)
//    }
//    coordinates(0) = new Coordinate(minX, minY)
//    coordinates(1) = new Coordinate(minX, maxY)
//    coordinates(2) = new Coordinate(maxX, maxY)
//    coordinates(3) = new Coordinate(maxX, minY)
//    coordinates(4) = coordinates(0)
//    val geometryFactory = new GeometryFactory()
//    buffer(0) = geometryFactory.createPolygon(coordinates)
//  }
//
//  override def merge(buffer1: MutableAggregationBuffer, buffer2: Row): Unit = {
//    val leftEnvelope = buffer1.getAs[Geometry](0).getEnvelopeInternal
//    val rightEnvelope = buffer2.getAs[Geometry](0).getEnvelopeInternal
//    val coordinates: Array[Coordinate] = new Array[Coordinate](5)
//    var minX = 0.0
//    var minY = 0.0
//    var maxX = 0.0
//    var maxY = 0.0
//    if (leftEnvelope.getMinX == -999999999) {
//      minX = rightEnvelope.getMinX
//      minY = rightEnvelope.getMinY
//      maxX = rightEnvelope.getMaxX
//      maxY = rightEnvelope.getMaxY
//    } else if (rightEnvelope.getMinX == -999999999) {
//      minX = leftEnvelope.getMinX
//      minY = leftEnvelope.getMinY
//      maxX = leftEnvelope.getMaxX
//      maxY = leftEnvelope.getMaxY
//    } else {
//      minX = Math.min(leftEnvelope.getMinX, rightEnvelope.getMinX)
//      minY = Math.min(leftEnvelope.getMinY, rightEnvelope.getMinY)
//      maxX = Math.max(leftEnvelope.getMaxX, rightEnvelope.getMaxX)
//      maxY = Math.max(leftEnvelope.getMaxY, rightEnvelope.getMaxY)
//    }
//    coordinates(0) = new Coordinate(minX, minY)
//    coordinates(1) = new Coordinate(minX, maxY)
//    coordinates(2) = new Coordinate(maxX, maxY)
//    coordinates(3) = new Coordinate(maxX, minY)
//    coordinates(4) = coordinates(0)
//    val geometryFactory = new GeometryFactory()
//    buffer1(0) = geometryFactory.createPolygon(coordinates)
//  }
//
//  override def evaluate(buffer: Row): Any = buffer.getAs[Geometry](0)
//}
