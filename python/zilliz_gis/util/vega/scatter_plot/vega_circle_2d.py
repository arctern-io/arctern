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
from zilliz_gis.util.vega.scatter_plot.vega_scatter_plot import VegaScatterPlot
from zilliz_gis.util.vega.vega_node import (RootMarks, Root, Description, Data,
                                            Width, Height, Scales)

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

        def __init__(self, shape: Value, stroke: Value, strokeWidth: Value, opacity: Value):
            if not (isinstance(shape.v, str) and isinstance(strokeWidth.v, int) and
                    isinstance(stroke.v, str) and isinstance(opacity.v, float)):
                # TODO error log here
                print("illegal")
                assert 0
            self._shape = shape
            self._stroke = stroke
            self._strokeWidth = strokeWidth
            self._opacity = opacity

        def to_dict(self):
            dic = {
                "enter": {
                    "shape": self._shape.to_dict(),
                    "stroke": self._stroke.to_dict(),
                    "strokeWidth": self._strokeWidth.to_dict(),
                    "opacity": self._opacity.to_dict()
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

class VegaCircle2d(VegaScatterPlot):
    def __init__(self, width: int, height: int, mark_size: int, mark_color: str, opacity: float):
        VegaScatterPlot.__init__(self, width, height)
        self._mark_size = mark_size
        self._mark_color = mark_color
        self._opacity = opacity

    def build(self):
        description = Description(desc="circle_2d")
        data = Data(name="data", url="/data/data.csv")
        domain1 = Scales.Scale.Domain(data="data", field="c0")
        domain2 = Scales.Scale.Domain(data="data", field="c1")
        scale1 = Scales.Scale("x", "linear", domain1)
        scale2 = Scales.Scale("y", "linear", domain2)
        scales = Scales([scale1, scale2])
        encode = Marks.Encode(shape=Marks.Encode.Value("circle"),
                              stroke=Marks.Encode.Value(self._mark_color),
                              strokeWidth=Marks.Encode.Value(self._mark_size),
                              opacity=Marks.Encode.Value(self._opacity))
        marks = Marks(encode)
        root = Root(Width(self._width), Height(self._height), description,
                    data, scales, marks)

        root_json = json.dumps(root.to_dict(), indent=2)
        return root_json
