import pyarrow
import zilliz_gis
import pandas
import numpy
import pytest

def test_point_map():
    x_data = []
    y_data = []

    # y = 150
    for i in range(100, 200):
        x_data.append(i)
        y_data.append(150)

    # y = x - 50
    for i in range(100, 200):
        x_data.append(i)
        y_data.append(i - 50)

    # y = 50
    for i in range(100, 200):
        x_data.append(i)
        y_data.append(50)

    arr_x = pyarrow.array(x_data, type='uint32')
    arr_y = pyarrow.array(y_data, type='uint32')

    curve_z = zilliz_gis.point_map(arr_x, arr_y)
    curve_z = curve_z.buffers()[1].to_pybytes()
    print(curve_z)

# def test_heat_map():
#     curve_z = zilliz_gis.point_map(arr_x, arr_y)