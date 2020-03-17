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
from arctern.util.vega.scatter_plot.vega_scatter_plot import VegaScatterPlot
from arctern.util.vega.vega_node import (RootMarks, Root, Description, Data,
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

        def __init__(self, bounding_box_min: Value, bounding_box_max: Value, shape: Value, stroke: Value,
                     stroke_width: Value, opacity: Value, coordinate_system: Value):
            if not (isinstance(bounding_box_min.v, str)
                    and isinstance(bounding_box_max.v, str)
                    and isinstance(shape.v, str)
                    and isinstance(stroke_width.v, int)
                    and isinstance(stroke.v, str)
                    and isinstance(opacity.v, float)
                    and isinstance(coordinate_system.v, str)):
                # TODO error log here
                print("illegal")
                assert 0
            self._bounding_box_min = bounding_box_min
            self._bounding_box_max = bounding_box_max
            self._shape = shape
            self._stroke = stroke
            self._stroke_width = stroke_width
            self._opacity = opacity
            self._coordinate_system = coordinate_system

        def to_dict(self):
            dic = {
                "enter": {
                    "bounding_box_min": self._bounding_box_min.to_dict(),
                    "bounding_box_max": self._bounding_box_max.to_dict(),
                    "shape": self._shape.to_dict(),
                    "stroke": self._stroke.to_dict(),
                    "strokeWidth": self._stroke_width.to_dict(),
                    "opacity": self._opacity.to_dict(),
                    "coordinate_system": self._coordinate_system.to_dict()
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
    def __init__(self, width: int, height: int, bounding_box_min: str, bounding_box_max: str,
                 mark_size: int, mark_color: str, opacity: float, coordinate_system: str):
        VegaScatterPlot.__init__(self, width, height)
        self._bounding_box_min = bounding_box_min
        self._bounding_box_max = bounding_box_max
        self._mark_size = mark_size
        self._mark_color = mark_color
        self._opacity = opacity
        self._coordinate_system = coordinate_system

    def build(self):
        description = Description(desc="circle_2d")
        data = Data(name="data", url="/data/data.csv")
        domain1 = Scales.Scale.Domain(data="data", field="c0")
        domain2 = Scales.Scale.Domain(data="data", field="c1")
        scale1 = Scales.Scale("x", "linear", domain1)
        scale2 = Scales.Scale("y", "linear", domain2)
        scales = Scales([scale1, scale2])
        encode = Marks.Encode(bounding_box_min=Marks.Encode.Value(self._bounding_box_min),
                              bounding_box_max=Marks.Encode.Value(self._bounding_box_max),
                              shape=Marks.Encode.Value("circle"),
                              stroke=Marks.Encode.Value(self._mark_color),
                              stroke_width=Marks.Encode.Value(self._mark_size),
                              opacity=Marks.Encode.Value(self._opacity),
                              coordinate_system=Marks.Encode.Value(self._coordinate_system))
        marks = Marks(encode)
        root = Root(Width(self._width), Height(self._height), description,
                    data, scales, marks)

        root_json = json.dumps(root.to_dict(), indent=2)
        return root_json
