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

import org.apache.spark.sql.catalyst.util.{ArrayData, GenericArrayData}
import org.apache.spark.sql.types.{ArrayType, ByteType, DataType, UserDefinedType}
import org.locationtech.jts.index.SpatialIndex
import java.io.{ByteArrayInputStream, ByteArrayOutputStream}

import com.esotericsoftware.kryo.Kryo
import com.esotericsoftware.kryo.io.{Input, Output}
import org.apache.spark.sql.catalyst.util.ArrayData

object IndexSerializer {
  def serialize(index: SpatialIndex): Array[Byte] = {
    val out = new ByteArrayOutputStream()
    val kryo = new Kryo()
    val output = new Output(out)
    kryo.writeObject(output, index)
    output.close()
    out.toByteArray
  }

  def deserialize(values: ArrayData): SpatialIndex = {
    val in = new ByteArrayInputStream(values.toByteArray())
    val kryo = new Kryo()
    val input = new Input(in)
    val spatialIndex = kryo.readObject(input, classOf[SpatialIndex])
    input.close()
    spatialIndex.asInstanceOf[SpatialIndex]
  }

}

class IndexUDT extends UserDefinedType[SpatialIndex] {
  override def sqlType: DataType = ArrayType(ByteType, containsNull = false)

  override def serialize(obj: SpatialIndex): GenericArrayData = new GenericArrayData(IndexSerializer.serialize(obj))

  override def deserialize(datum: Any): SpatialIndex = {
    datum match {
      case values: ArrayData => IndexSerializer.deserialize(values)
      case null => null
    }
  }

  override def userClass: Class[SpatialIndex] = classOf[SpatialIndex]
}
