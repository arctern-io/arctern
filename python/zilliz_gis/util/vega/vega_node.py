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

from typing import List
import abc

class Width:
    """
        Top-Level Vega Specification Property: Width
    """
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

class Height:
    """
        Top-Level Vega Specification Property: Height
    """

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

class Description:

    """
        Top-Level Vega Specification Property: Description
        oneOf(enum('icon_2d', 'circle_2d', 'multi_color_circles_2d', 'weighted_color_circles_2d',
    '   building_weighted_2d', 'heatmap_2d', 'get_building_shape'))
    """
    render_type = {"icon_2d", "circle_2d", "multi_color_circles_2d", "weighted_color_circles_2d",
                   "building_weighted_2d", "heat_map_2d", "get_building_shape"}

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

class Data:
    """
        Top-Level Vega Specification Property: Data
    """
    def __init__(self, name: str, url: str):
        self._name = name
        self._url = url

    def to_dict(self):
        dic = [{
            "name": self._name,
            "url": self._url
        }]
        return dic

class Scales:

    """
        Top-Level Vega Specification Property: Scales
    """
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

        def __init__(self, name: str, scale_type: str, domain: Domain):
            self._name = name
            self._type = scale_type
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

class RootMarks:
    """
        Top-Level Vega Specification Property: Marks
    """
    @abc.abstractmethod
    def to_dict(self):
        pass

class Root:
    """
        Vega Root
    """
    def __init__(self, width: Width, height: Height, description: Description,
                 data: Data, scales: Scales, marks: RootMarks):
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
