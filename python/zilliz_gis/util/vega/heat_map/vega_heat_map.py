from zilliz_gis.util.vega.vega_node import *

import json

"""
Top-Level Vega Specification Property: Marks
"""
class Marks(RootMarks):
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
            if not (type(map_scale.v) == float):
                # TODO error log here
                print("illegal")
                assert 0
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

class VegaHeatMap(object):
    def __init__(self, width: int, height: int, map_scale: float):
        self._width = width
        self._height = height
        self._map_scale = map_scale

    def build(self):
        description = Description(desc="heat_map_2d")
        data = Data(name="data", url="/data/data.csv")
        domain1 = Scales.Scale.Domain(data="data", field="c0")
        domain2 = Scales.Scale.Domain(data="data", field="c1")
        scale1 = Scales.Scale(name="x", type="linear", domain=domain1)
        scale2 = Scales.Scale(name="y", type="linear", domain=domain2)
        scales = Scales([scale1, scale2])
        encode = Marks.Encode(map_scale=Marks.Encode.Value(self._map_scale))
        marks = Marks(encode)
        root = Root(width=Width(self._width), height=Height(self._height), description=description,
                    data=data, scales=scales, marks=marks)

        root_json = json.dumps(root.to_dict(), indent=2)
        return root_json