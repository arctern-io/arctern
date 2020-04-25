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
from arctern.util.vega.vega_node import (Width, Height, Description, Data,
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

        def __init__(self, bounding_box: Value,
                     map_zoom_level: Value,
                     coordinate_system: Value,
                     aggregation_type: Value):
            if not (isinstance(bounding_box.v, list)
                    and isinstance(map_zoom_level.v, float)
                    and isinstance(coordinate_system.v, str)
                    and isinstance(aggregation_type.v, str)):
                # TODO error log here
                assert 0, "illegal"
            self._bounding_box = bounding_box
            self._map_zoom_level = map_zoom_level
            self._coordinate_system = coordinate_system
            self._aggregation_type = aggregation_type

        def to_dict(self):
            dic = {
                "enter": {
                    "bounding_box": self._bounding_box.to_dict(),
                    "map_zoom_level": self._map_zoom_level.to_dict(),
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


class VegaHeatMap:
    def __init__(self, width: int, height: int, bounding_box: list,
                 map_zoom_level: float, coordinate_system: str, aggregation_type: str):
        self._width = width
        self._height = height
        self._bounding_box = bounding_box
        self._map_zoom_level = float(map_zoom_level)
        self._coordinate_system = coordinate_system
        self._aggregation_type = aggregation_type

    def build(self):
        description = Description(desc="heat_map_2d")
        data = Data(name="data", url="/data/data.csv")
        domain1 = Scales.Scale.Domain("data", "c0")
        domain2 = Scales.Scale.Domain("data", "c1")
        scale1 = Scales.Scale("x", "linear", domain1)
        scale2 = Scales.Scale("y", "linear", domain2)
        scales = Scales([scale1, scale2])
        encode = Marks.Encode(bounding_box=Marks.Encode.Value(self._bounding_box),
                              map_zoom_level=Marks.Encode.Value(self._map_zoom_level),
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
