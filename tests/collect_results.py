import random
import os
import inspect
import sys
import shutil
import glob
from util import *


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
            shutil.copyfile(file_name[0],
                            os.path.join(arctern_result, y + '.csv'))
        else:
            print('file [%s] not exist' % file_name[0])


# def get_sort_zgis(file_path):
#     with open(file_path, 'r') as f:
#         lines = f.readlines()
#         run_funcs = [x.strip().lower() for x in lines]
#         run_funcs.sort()
#         for x in run_funcs:
#             print(x)

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # collect_results(os.path.join(base_dir, './config.txt'), arctern_result)
    collect_results()
