import pyarrow
from zilliz_gis import point_map

def _savePNG(data, png_name):
    try:
        imageData = data
    except BaseException as e:
        pass
    # save result png as fixed png
    else:
        with open(png_name, "wb") as tmp_file:
            tmp_file.write(imageData)


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

curve_z = point_map(arr_x, arr_y)
curve_z = curve_z.buffers()[1].to_pybytes()

_savePNG(curve_z, "/tmp/curve_z.png")
