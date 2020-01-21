from pyspark.sql import SparkSession
import pandas
import pyarrow
from zilliz_gis import point_map

# run this script via
# /path/to/spark-submit test_pointmap.py

def _savePNG(data, png_name):
    try:
        imageData = data
    except BaseException as e:
        pass
    # save result png as fixed png
    else:
        with open(png_name, "wb") as tmp_file:
            tmp_file.write(imageData)

def gen_curve_z():
    with open("/tmp/z_curve.json", "w") as file:
        # y = 150
        for i in range(100, 200):
            line = '{"x": %d, "y": %d}\n' % (i, 150)
            file.write(line)

        # y = x - 50
        for i in range(100, 200):
            line = '{"x": %d, "y": %d}\n' % (i, i - 50)
            file.write(line)

        # y = 50
        for i in range(100, 200):
            line = '{"x": %d, "y": %d}\n' % (i, 50)
            file.write(line)

def run_curve_z(spark):
    curve_z_df = spark.read.json("/tmp/z_curve.json").cache()

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


if __name__ == "__main__":
    spark = SparkSession \
        .builder \
        .appName("Python TestPointmap") \
        .getOrCreate()

    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")

    gen_curve_z()
    run_curve_z(spark)

    spark.stop()




