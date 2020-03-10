import random
import os
import inspect
import sys
import shutil

curves = ['CIRCULARSTRING', 'COMPOUNDCURVE', 'MULTICURVE', 'CURVEPOLYGON', 'MULTISURFACE']

def to_disk(file_path, content):

    with open(file_path, 'w') as f:
        if isinstance(content, list):
            for line in content:
                f.writelines(line)


def ilnormal(geom):
    geom = geom.strip().upper()
    for x in curves:
        if geom.startswith(x):
            return True
        else:
            continue
    
    return False


def split_data(file_path):
    wholename = os.path.abspath(file_path)
    basename = os.path.basename(file_path)
    basedir = os.path.split(wholename)[0]
    file_name = os.path.splitext(basename)[0]
    file_ext = os.path.splitext(basename)[1]

    normal_arr = []
    ilnormail_arr = []
    with open(file_path, 'r') as f:
        lines = f.readlines()[1:]
        ds = [x.strip().upper().split('|') for x in lines]
        for line in ds:
            if len(line) == 1:
                if ilnormal(line[0].strip()):
                    ilnormail_arr.append(line[0].strip() + '\n')
                else:
                    normal_arr.append(line[0].strip() + '\n')
            elif len(line) == 2:
                if ilnormal(line[0].strip()) or ilnormal(line[1].strip()):
                    ilnormail_arr.append(line[0].strip() + '|' + line[1].strip() + '\n')
                else:
                    normal_arr.append(line[0].strip() + '|' + line[1].strip() + '\n')

    if len(normal_arr) > 0:
        if '|' in normal_arr[0]:
            normal_arr.insert(0, 'left|right\n')
        else:
            normal_arr.insert(0, 'geos\n')
        # to_disk(os.path.join(basedir, file_name + '_normal' + file_ext), normal_arr)
        to_disk(os.path.join(basedir, basename), normal_arr)

    if len(ilnormail_arr) > 0:
        if '|' in ilnormail_arr[0]:
            ilnormail_arr.insert(0, 'left|right\n')
        else:
            ilnormail_arr.insert(0, 'geos\n')
        to_disk(os.path.join(basedir, file_name + '_ilnormal' + file_ext), ilnormail_arr)
       

def split_all(folder):
    for f in os.listdir(folder):
        if os.path.isfile(os.path.join(folder, f)):
            print(f)
            if f.endswith('.csv'):
                if not ('_normal' in f or '_ilnormal' in f):
                    split_data(os.path.join(folder, f))
                

if __name__ == '__main__':
    # split_data('./data/area.csv')
    # split_data('./data/crosses.csv')
    split_all('./data')
