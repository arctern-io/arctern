import random
import os
import inspect
import sys
import shutil           
import glob


def collect_results(file_path, results_dir):
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    with open(file_path, 'r') as f:
        cs = f.readlines()
        cs = [x.strip() for x in cs if not x.startswith('#')]
        names = [x.strip().split('=')[0] for x in cs]
        table_names = [x.strip().split('=')[1] for x in cs]
        # for x in names:
        #     print(x)
        # print('------------------')
        # for x in table_names:
        #     print(x)

    base_dir = '/tmp/results'
    for x, y in zip(table_names, names):
        print(x, y)
        target = os.path.join(base_dir, x, '*.csv')
        file_name = glob.glob(target)
        print(file_name)
        print(file_name[0])
        if os.path.isfile(file_name[0]):
            shutil.copyfile(file_name[0], os.path.join(results_dir, y +'.csv'))
        else:
            print('file [%s] not exist' % file_name[0])


def get_sort_zgis(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        run_funcs = [x.strip().lower() for x in lines]
        run_funcs.sort()
        for x in run_funcs:
            print(x)

if __name__ == '__main__':
    # collect_results('./config.txt', './arctern_results')
    collect_results('./config.txt', '/tmp/arctern_results')
