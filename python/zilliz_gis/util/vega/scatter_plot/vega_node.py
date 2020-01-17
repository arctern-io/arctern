from typing import List

"""
Top-Level Vega Specification Property: Width
"""
class Width:
    def __init__(self, width: int):
        if width <= 0:
            # TODO error log here
            print("illegal")
            assert 0
        self._width = width

    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, width):
        self._width = width

"""
Top-Level Vega Specification Property: Height
"""
class Height:
    def __init__(self, height: int):
        if height <= 0:
            # TODO error log here
            print("illegal")
            assert 0
        self._height = height

    @property
    def height(self):
        return self._height

    @height.setter
    def height(self, height):
        self._height = height

"""
Top-Level Vega Specification Property: Description
oneOf(enum('icon_2d', 'circle_2d', 'multi_color_circles_2d', 'weighted_color_circles_2d', 
    'building_weighted_2d', 'heatmap_2d', 'get_building_shape'))
"""
class Description:
    render_type = {"icon_2d", "circle_2d", "multi_color_circles_2d", "weighted_color_circles_2d",
                   "building_weighted_2d", "heatmap_2d", "get_building_shape"}

    def __init__(self, desc: str):
        if desc not in self.render_type:
            # TODO error log here
            print("illegal")
            assert 0
        self._description = desc

    @property
    def description(self):
        return self._description

    @description.setter
    def description(self, description):
        self._description = description

"""
Top-Level Vega Specification Property: Data
"""
class Data:
    def __init__(self, name: str, url: str):
        self._name = name
        self._url = url

    def to_dict(self):
        dic = [{
            "name": self._name,
            "url": self._url
        }]
        return dic

"""
Top-Level Vega Specification Property: Scales
"""
class Scales:
    class Scale:
        class Domain:
            def __init__(self, data: str, field: str):
                self._data = data
                self._field = field

            def to_dict(self):
                dic = {
                    "data": self._data,
                    "field": self._field
                }
                return dic

        def __init__(self, name: str, type: str, domain: Domain):
            self._name = name
            self._type = type
            self._domain = domain

        def to_dict(self):
            dic = {
                "name": self._name,
                "type": self._type,
                "domain": self._domain.to_dict()
            }
            return dic

    def __init__(self, scales: List[Scale]):
        self._scales = scales

    def to_dict(self):
        dic = []
        for s in self._scales:
            dic.append(s.to_dict())
        return dic

"""
Top-Level Vega Specification Property: Marks
"""
class Marks:
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

"""
Vega Root
"""
class Root:
    def __init__(self, width: Width, height: Height, description: Description,
                 data: Data, scales: Scales, marks: Marks):
        self._width = width
        self._height = height
        self._description = description
        self._data = data
        self._scales = scales
        self._marks = marks

    def to_dict(self):
        dic = {
            "width": self._width.width,
            "height": self._height.height,
            "description": self._description.description,
            "data": self._data.to_dict(),
            "scales": self._scales.to_dict(),
            "marks": self._marks.to_dict()
        }
        return dic
