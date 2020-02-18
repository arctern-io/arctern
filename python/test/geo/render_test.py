import pyarrow
import zilliz_gis
import pandas
import numpy
import pytest

from zilliz_gis.util.vega.scatter_plot.vega_circle_2d import VegaCircle2d
from zilliz_gis.util.vega.heat_map.vega_heat_map import VegaHeatMap

def _savePNG(data, png_name):
    try:
        imageData = data
    except BaseException as e:
        pass
    # save result png as fixed png
    else:
        with open(png_name, "wb") as tmp_file:
            tmp_file.write(imageData)

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

    vega_circle2d = VegaCircle2d(300, 200, 30, "#ff0000", 0.5)
    vega_json = vega_circle2d.build()

    curve_z = zilliz_gis.point_map(arr_x, arr_y, vega_json.encode('utf-8'))
    curve_z = curve_z.buffers()[1].to_pybytes()

    _savePNG(curve_z, "/tmp/curve_z.png")

def test_heat_map():
    x_data = []
    y_data = []
    c_data = []

    for i in range(0, 5):
        x_data.append(i + 50)
        y_data.append(i + 50)
        c_data.append(i + 50)

    arr_x = pyarrow.array(x_data, type='uint32')
    arr_y = pyarrow.array(y_data, type='uint32')
    arr_c = pyarrow.array(y_data, type='uint32')

    vega_heat_map = VegaHeatMap(300, 200, 10.0)
    vega_json = vega_heat_map.build()

    heat_map = zilliz_gis.heat_map(arr_x, arr_y, arr_c, vega_json.encode('utf-8'))
    heat_map = heat_map.buffers()[1].to_pybytes()

    _savePNG(heat_map, "/tmp/test_heat_map.png")