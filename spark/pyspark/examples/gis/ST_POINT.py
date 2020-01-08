import pyarrow
import pandas
import numpy

import random

if __name__ == "__main__":
    xlist = [random.random() * 100 for i in range(1000)]
    ylist = [random.random() * 100 for i in range(1000)]

    arr_x = pyarrow.array(xlist, type=pyarrow.float64())
    arr_y = pyarrow.array(ylist, type=pyarrow.float64())

    from zillizgis import ST_POINT
    arr_point = ST_POINT(arr_x, arr_y)

    print(arr_x)
    print(arr_y)
    print(arr_point)
