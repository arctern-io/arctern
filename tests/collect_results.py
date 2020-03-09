import random
import os
import inspect
import sys
import shutil

def get_functions(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        run_funcs = [x.strip() for x in lines if x.strip().startswith('def run')]
        for x in run_funcs:
            print(x.split('(')[0].split(' ')[1])

def get_sqls(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        run_funcs = [x.strip().lower() for x in lines if x.strip().startswith('sql')]
        for x in run_funcs:
            print(x.split('=')[1].strip()[1:-1])


def get_sqls_and_data(file_path):
    d = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()
        run_funcs = [x.strip().lower() for x in lines if x.strip().startswith('sql =')]
        run_datas = [x.strip().lower() for x in lines if x.strip().startswith('data =')]
        
        for x, y in zip(run_funcs, run_datas):
            sql = x.split('=')[1].strip()[1:-1]
            file_name = y.split('=')[1].strip()[1:-1]
            with open('./data/' + file_name, 'r') as ff:
                content = ff.readlines()
            d[sql] = content
    
    # k = d.keys()[0]
    # print(d[k])
    with open('delt_file.txt', 'w') as f:
        for k, v in d.items():
            f.writelines(k + '\n')
            f.writelines(v)
            f.writelines('\n')

def get_sqls_and_table_names(file_path):
    d = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()
        run_funcs = [x.strip() for x in lines if x.strip().startswith('def run')]
        run_datas = [x.strip().lower() for x in lines if x.strip().startswith('table_name =')]
        
        for x, y in zip(run_funcs, run_datas):
            sql = x.split('(')[0].split(' ')[1][9:] + '_udf'
            
            table_name = y.split('=')[1].strip()[1:-1]
            print(sql + '=' + table_name)
            # print(table_name)
            
import glob
def collect_results(file_path, results_dir):
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
        target = os.path.join(base_dir, x, '*.json')
        file_name = glob.glob(target)
        # file_name = glob.glob(target)
        print(file_name)
        print(file_name[0])
        if os.path.isfile(file_name[0]):
            shutil.copyfile(file_name[0], os.path.join(results_dir, y +'.json'))
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
