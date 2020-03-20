# import random
import os
# import inspect
import shutil
import glob
from util import arctern_result
from util import spark_result
from util import get_tests


def collect_results():
    """Collect spark results from different dirs."""
    if not os.path.isdir(arctern_result):
        os.makedirs(arctern_result)

    names, table_names, expects = get_tests()

    base_dir = spark_result
    for table_name, name in zip(table_names, names):
        target = os.path.join(base_dir, table_name, '*.csv')
        file_name = glob.glob(target)
        if os.path.isfile(file_name[0]):
            shutil.copyfile(file_name[0], os.path.join(
                arctern_result, name + '.csv'))
        else:
            print('file [%s] not exist' % file_name[0])


if __name__ == '__main__':
    collect_results()
