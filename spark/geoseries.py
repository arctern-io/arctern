# Copyright (C) 2019-2020 Zilliz. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import databricks.koalas as ks
ks.set_option('compute.ops_on_diff_frames', True)
import scala_wrapper


class GeoSeries(ks.Series):
    @classmethod
    def point(self, x, y):
        from pyspark.sql.functions import col, lit
        series1 = ks.Series(x, name = "col1", dtype=int)
        series2 = ks.Series(y, dtype=int)
        kdf2 = ks.DataFrame(data = series1)
        kdf2['col2'] =  series2
        sdf = kdf2.to_spark()

        ret = sdf.select(scala_wrapper.st_point("col1", "col2")) 
        ret.show()
        # to do 
        # return koalas Series

x = list(range(100000))
y = list(range(100000))
z = GeoSeries.point(x, y)
