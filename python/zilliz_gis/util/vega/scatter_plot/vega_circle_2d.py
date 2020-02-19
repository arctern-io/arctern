from zilliz_gis.util.vega.scatter_plot.vega_scatter_plot import VegaScatterPlot
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

        def __init__(self, shape: Value, stroke: Value, strokeWidth: Value, opacity: Value):
            if not (type(shape.v) == str and type(strokeWidth.v) == int and
                    type(stroke.v) == str and type(opacity.v) == float):
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
        scale1 = Scales.Scale(name="x", type="linear", domain=domain1)
        scale2 = Scales.Scale(name="y", type="linear", domain=domain2)
        scales = Scales([scale1, scale2])
        encode = Marks.Encode(shape=Marks.Encode.Value("circle"),
                              stroke=Marks.Encode.Value(self._mark_color),
                              strokeWidth=Marks.Encode.Value(self._mark_size),
                              opacity=Marks.Encode.Value(self._opacity))
        marks = Marks(encode)
        root = Root(width=Width(self._width), height=Height(self._height), description=description,
                    data=data, scales=scales, marks=marks)

        root_json = json.dumps(root.to_dict(), indent=2)
        return root_json
