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

        def __init__(self, bounding_box: Value, color_style: Value, ruler: Value, opacity: Value):
            if not (isinstance(bounding_box.v, list) and isinstance(color_style.v, str) and
                    isinstance(ruler.v, list) and isinstance(opacity.v, float)):
                # TODO error log here
                assert 0, "illegal"
            self._bounding_box = bounding_box
            self._color_style = color_style
            self._ruler = ruler
            self._opacity = opacity

        def to_dict(self):
            dic = {
                "enter": {
                    "bounding_box": self._bounding_box.to_dict(),
                    "color_style": self._color_style.to_dict(),
                    "ruler": self._ruler.to_dict(),
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

class VegaChoroplethMap:
    def __init__(self, width: int, height: int,
                 bounding_box: list, color_style: str,
                 ruler: list, opacity: float):
        self._width = width
        self._height = height
        self._bounding_box = bounding_box
        self._color_style = color_style
        self._ruler = ruler
        self._opacity = opacity

    def build(self):
        description = Description(desc="building_weighted_2d")
        data = Data(name="data", url="/data/data.csv")
        domain = Scales.Scale.Domain("data", "c0")
        scale = Scales.Scale("building", "linear", domain)
        scales = Scales([scale])
        encode = Marks.Encode(bounding_box=Marks.Encode.Value(self._bounding_box),
                              color_style=Marks.Encode.Value(self._color_style),
                              ruler=Marks.Encode.Value(self._ruler),
                              opacity=Marks.Encode.Value(self._opacity))
        marks = Marks(encode)
        root = Root(Width(self._width), Height(self._height), description,
                    data, scales, marks)

        root_json = json.dumps(root.to_dict(), indent=2)
        return root_json
