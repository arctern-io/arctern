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

import json
from zilliz_gis.util.vega.vega_node import (Width, Height, Description, Data,
                                            Scales, RootMarks, Root)


class Marks(RootMarks):
    """
        Top-Level Vega Specification Property: Marks
    """

    class Encode:
        class Value:
            def __init__(self, v: int or float or str):
                self.v = v

            def to_dict(self):
                dic = {
                    "value": self.v
                }
                return dic

        def __init__(self, map_scale: Value):
            if not isinstance(map_scale.v, float):
                # TODO error log here
                assert 0, "illegal"
            self._map_scale = map_scale

        def to_dict(self):
            dic = {
                "enter": {
                    "map_scale": self._map_scale.to_dict()
                }
            }
            return dic

    def __init__(self, encode: Encode):
        self.encode = encode

    def to_dict(self):
        dic = [{
            "encode": self.encode.to_dict()
        }]
        return dic

class VegaHeatMap():
    def __init__(self, width: int, height: int, map_scale: float):
        self._width = width
        self._height = height
        self._map_scale = map_scale

    def build(self):
        description = Description(desc="heat_map_2d")
        data = Data(name="data", url="/data/data.csv")
        domain1 = Scales.Scale.Domain("data", "c0")
        domain2 = Scales.Scale.Domain("data", "c1")
        scale1 = Scales.Scale("x", "linear", domain1)
        scale2 = Scales.Scale("y", "linear", domain2)
        scales = Scales([scale1, scale2])
        encode = Marks.Encode(map_scale=Marks.Encode.Value(self._map_scale))
        marks = Marks(encode)
        root = Root(Width(self._width), Height(self._height), description,
                    data, scales, marks)

        root_json = json.dumps(root.to_dict(), indent=2)
        return root_json
