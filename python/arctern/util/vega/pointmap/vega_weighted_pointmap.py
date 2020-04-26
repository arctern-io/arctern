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
from arctern.util.vega.pointmap.vega_scatter_plot import VegaScatterPlot
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

        def __init__(self, bounding_box: Value, shape: Value, color_gradient: Value,
                     color_bound: Value, size_bound: Value, opacity: Value,
                     coordinate_system: Value, aggregation_type: Value):
            if not (isinstance(bounding_box.v, list)
                    and isinstance(shape.v, str)
                    and isinstance(color_gradient.v, list)
                    and isinstance(color_bound.v, list)
                    and isinstance(size_bound.v, list)
                    and isinstance(opacity.v, float)
                    and isinstance(coordinate_system.v, str)
                    and isinstance(aggregation_type.v, str)):
                # TODO error log here
                print("illegal")
                assert 0
            self._bounding_box = bounding_box
            self._shape = shape
            self._color_gradient = color_gradient
            self._color_bound = color_bound
            self._size_bound = size_bound
            self._opacity = opacity
            self._coordinate_system = coordinate_system
            self._aggregation_type = aggregation_type

        def to_dict(self):
            dic = {
                "enter": {
                    "bounding_box": self._bounding_box.to_dict(),
                    "shape": self._shape.to_dict(),
                    "color_gradient": self._color_gradient.to_dict(),
                    "color_bound": self._color_bound.to_dict(),
                    "size_bound": self._size_bound.to_dict(),
                    "opacity": self._opacity.to_dict(),
                    "coordinate_system": self._coordinate_system.to_dict(),
                    "aggregation_type": self._aggregation_type.to_dict()
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


class VegaWeightedPointMap(VegaScatterPlot):
    def __init__(self, width: int, height: int, bounding_box: list,
                 color_gradient: list, color_bound: list, size_bound: list, opacity: float,
                 coordinate_system: str, aggregation_type: str):

        VegaScatterPlot.__init__(self, width, height)
        self._bounding_box = bounding_box
        self._color_gradient = color_gradient
        self._color_bound = color_bound
        self._size_bound = size_bound
        self._opacity = opacity
        self._coordinate_system = coordinate_system
        self._aggregation_type = aggregation_type

    def build(self):
        description = Description(desc="circle_2d")
        data = Data(name="data", url="/data/data.csv")
        domain1 = Scales.Scale.Domain(data="data", field="c0")
        domain2 = Scales.Scale.Domain(data="data", field="c1")
        scale1 = Scales.Scale("x", "linear", domain1)
        scale2 = Scales.Scale("y", "linear", domain2)
        scales = Scales([scale1, scale2])
        encode = Marks.Encode(bounding_box=Marks.Encode.Value(self._bounding_box),
                              shape=Marks.Encode.Value("circle"),
                              color_gradient=Marks.Encode.Value(self._color_gradient),
                              color_bound=Marks.Encode.Value(self._color_bound),
                              size_bound=Marks.Encode.Value(self._size_bound),
                              opacity=Marks.Encode.Value(self._opacity),
                              coordinate_system=Marks.Encode.Value(self._coordinate_system),
                              aggregation_type=Marks.Encode.Value(self._aggregation_type))
        marks = Marks(encode)
        root = Root(Width(self._width), Height(self._height), description,
                    data, scales, marks)

        root_json = json.dumps(root.to_dict(), indent=2)
        return root_json

    def coor(self):
        return self._coordinate_system

    def bounding_box(self):
        return self._bounding_box

    def height(self):
        return self._height

    def width(self):
        return self._width

    def aggregation_type(self):
        return self._aggregation_type
