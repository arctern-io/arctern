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

import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.arctern._
import org.scalatest.{BeforeAndAfterAll, FunSuite}

class AdapterTest extends FunSuite with BeforeAndAfterAll {

  var spark: SparkSession = _

  override def beforeAll {
    Logger.getLogger("org").setLevel(Level.WARN)
    spark = SparkSession.builder()
      .master("local[*]")
      .appName("arctern scala test")
      .getOrCreate()

    UdtRegistratorWrapper.registerUDT()
    UdfRegistrator.register(spark)

  }

  //  override def afterAll {
  //    spark.stop()
  //  }
}