import random
import os
import inspect
import shutil
import glob
from util import arctern_result
from util import get_tests


def collect_results():
    if not os.path.isdir(arctern_result):
        os.makedirs(arctern_result)

    names, table_names, expects = get_tests()

    base_dir = spark_result
    for x, y in zip(table_names, names):
        # print(x, y)
        target = os.path.join(base_dir, x, '*.csv')
        file_name = glob.glob(target)
        if os.path.isfile(file_name[0]):
            shutil.copyfile(file_name[0], os.path.join(arctern_result, y + '.csv'))
        else:
            print('file [%s] not exist' % file_name[0])


if __name__ == '__main__':
    collect_results()

