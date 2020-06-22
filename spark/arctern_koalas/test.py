import databricks.koalas as ks
from databricks.koalas.series import _col

ks.set_option('compute.ops_on_diff_frames', True)
import scala_wrapper


def point(x, y):
    series1 = ks.Series(x, name="col1", dtype=int)
    series2 = ks.Series(y, dtype=int)
    kdf2 = ks.DataFrame(data=series1)
    kdf2['col2'] = series2
    sdf = kdf2.to_spark()

    ret = sdf.select(scala_wrapper.st_point("col1", "col2"))
    kdf = ret.to_koalas()
    s = _col(kdf)
    return s
    # to do
    # return koalas Series


x = list(range(100000))
y = list(range(100000))
z = point(x, y)
print(z)
