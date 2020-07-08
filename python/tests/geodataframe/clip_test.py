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

import arctern
from arctern import GeoDataFrame, GeoSeries


def test_clip():
    s3 = GeoSeries(["POLYGON ((2 1,3 1,3 2,2 2,2 1))",
                    "POLYGON ((-1 1, 1.5 1, 1.5 2, -1 2, -1 1))",
                    "POLYGON ((10 10, 20 10, 20 20, 10 20, 10 10))"])
    d1 = GeoDataFrame({"geo":s3})
    rst = arctern.clip(d1, "POLYGON ((1 1,1 2,2 2,2 1,1 1))", col="geo")
    assert len(rst) == 2
    assert isinstance(rst, GeoDataFrame)
