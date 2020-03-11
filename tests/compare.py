import random
import os
import inspect
import sys
import shutil           
import glob
# import pygeos
import shapely
from shapely import wkt

config_file = './config.txt'

geo_types = ['POLYGON', 'POINT', 'LINESTRING', 'CURVEPOLYGON']
geo_collection_types = ['MULTIPOLYGON', 'MULTIPOINT', 'MULTILINESTRING', 'GEOMETRYCOLLECTION']

def is_geometry(geo):
    geo = geo.strip().upper()

    for x in geo_types:
        if geo.startswith(x):
            return True
        else:
            continue
    
    return False

def is_geometrycollection(geo):
    geo = geo.strip().upper()

    for x in geo_collection_types:
        if geo.startswith(x):
            return True
        else:
            continue
    
    return False

def is_geometrytype(geo):
    geo = geo.strip().upper()

    arr = []
    arr.extend(geo_types)
    arr.extend(geo_collection_types)

    for x in arr:
        if x in geo:
            return True
        else:
            continue
    
    return False


UNIT = 0.0001
EPOCH = 1e-8

# def compare_geometry(x, y):
#     arct = pygeos.Geometry(x)
#     pgis = pygeos.Geometry(y)
#     dist = pygeos.measurement.hausdorff_distance(arct, pgis)
#     arct_length = pygeos.measurement.length(arct)
#     pgis_length = pygeos.measurement.length(pgis)
#     max_len = max(arct_length, pgis_length)
#     if dist > max_len * UNIT:
#         return False
#     else:
#         return True

def compare_geometry(x, y):
    arct = wkt.loads(x)
    pgis = wkt.loads(y)

    if x.upper().endswith('EMPTY') and y.upper().endswith('EMPTY'):
        return True
        
    result = arct.equals_exact(pgis, EPOCH)

    # if not result:
    #     print(arct, pgis)
    
    return result

def compare_geometrycollection(x, y):
    arct = wkt.loads(x)
    pgis = wkt.loads(y)
    result = arct.equals(pgis)

    # if not result:
    #     print(arct, pgis)
    
    return result

def compare_float(x, y):

    x = float(x)
    y = float(y)
    if abs((x - y)) <= EPOCH:
        return True
    else:
        # print(x, y)
        return False

def convert_str(strr):
    try:
        x = bool(strr)
        return x
    except:
        pass
    
    try:
        x = float(strr)
        return x
    except:
        pass

    return strr

def compare_one(result, expect):
    x = result[1]
    y = expect[1]
    print(type(x))
    print(type(y))
    print('result: %s' % str(x))
    print('expect: %s' % str(y))

    x = convert_str(x)
    y = convert_str(y)

    if y.strip() == 't':
        y = True
    elif y.strip() == 'f':
        y = False

    try:
        if isinstance(x, bool):
            flag = (x == y)
            if not flag:
                print(result[0], x, expect[0], y)
            return flag

        if isinstance(x, str):
            x = x.strip().upper()
            y = y.strip().upper()
            if is_geometry(x) and is_geometry(y):
                flag = compare_geometry(x, y)
                if not flag:
                    print(result[0], x, expect[0], y)
                return flag

            elif is_geometrycollection(x) and is_geometrycollection(y):
                flag = compare_geometrycollection(x, y)
                if not flag:
                    print(result[0], x, expect[0], y)
                return flag
            else:
                if is_geometrytype(x) and is_geometrytype(y):
                    flag = (x == y)
                    if not flag:
                        print(result[0], x, expect[0], y)
                    return flag

                print(result[0], x, expect[0], y)
                return False

        if isinstance(x, int) or isinstance(x, float):
            flag = compare_float(x, y)
            if not flag:
                print(result[0], x, expect[0], y)
            return flag
    except Exception as e:
        flag = False
    
    return flag


def compare_results(arctern_results, postgis_results):

    with open(arctern_results, 'r') as f:
        # arctern = f.readlines()
        arct_arr = []
        for (num, value) in enumerate(f, 1):
            if value.strip() != '':
                arct_arr.append((num, value.strip()))

    # arc = [list(eval(x.strip()).values())[0] for x in arctern if len(x.strip()) > 0]
    # print(arc)

    with open(postgis_results, 'r') as f:
        # postgis = f.readlines()
        pgis_arr = []
        for (num, value) in enumerate(f, 1):
            if value.strip() != '':
                pgis_arr.append((num, value.strip()))
    # pgis = [x.strip() for x in postgis if len(x.strip()) > 0]
    # print(pgis)
    
    flag = True

    if len(arct_arr) != len(pgis_arr):
        print('test result size: %s and expected result size: %s, NOT equal, check the two result files' % (len(arct_arr), len(pgis_arr)))
        return False

    for x, y in zip(arct_arr, pgis_arr):
        res = compare_one(x, y)
        flag = flag and res

    return flag

def parse(config_file):
    with open(config_file, 'r') as f:
        lines = f.readlines()
        xs = [x.strip().split('=') for x in lines if not x.strip().startswith('#')]
    return xs

# arc_result_dir = './arctern_results'
arc_result_dir = '/tmp/arctern_results'
pgis_result_dir = './expected/results'

def compare_all():
    configs = parse(config_file)
    if len(configs) == 0:
        print('No Arctern test results found, maybe something wrong in config file, please check: %s' % config_file)
        return 0

    for x in configs:
        
        # arctern_result = os.path.join(arc_result_dir, x[0] + '.json')
        arctern_result = os.path.join(arc_result_dir, x[0] + '.csv')
        postgis_result = os.path.join(pgis_result_dir, x[3] + '.out')
        print('Arctern test: %s, result compare started, test result: %s, expected result: %s' % (x[0], arctern_result, postgis_result))
        
        if not os.path.isfile(arctern_result):
            print('Arctern test: %s, result: FAILED, reason: %s' % (x[0], 'test result not found [%s]' % arctern_result))
            continue

        if not os.path.isfile(postgis_result):
            print('Arctern test: %s, result: FAILED, reason: %s' % (x[0], 'expected result not found [%s]' % postgis_result))
            continue

        res = compare_results(arctern_result, postgis_result)
        if res == True:
            print('Arctern test: %s, result: PASSED' % x[0])
        else:
            print('Arctern test: %s, result: FAILED' % x[0])


def update_quote(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
        update = content.replace(r'"', '')
    with open(file_path, 'w') as f:
        f.write(update)

def update_bool(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
        update = content.replace('true', 'True').replace('false', 'False')
    with open(file_path, 'w') as f:
        f.write(update)

def update_result():
    arr = ['run_test_st_issimple', 'run_test_st_intersects', 'run_test_st_contains', 'run_test_st_crosses', 'run_test_st_isvalid_1', 'run_test_st_overlaps', 'run_test_st_touches', 'run_test_st_within', 'run_test_st_equals_1', 'run_test_st_equals_2']
    configs = parse(config_file)
    if len(configs) == 0:
        print('No Arctern test results found, maybe something wrong in config file, please check: %s' % config_file)
        return 0

    for x in configs:
        arctern_result = os.path.join(arc_result_dir, x[0] + '.csv')
        if not os.path.isfile(arctern_result):
            print('Arctern test: %s, result: FAILED, reason: %s' % (x[0], 'test result not found [%s]' % arctern_result))
            continue

        if x[0] not in arr:
            update_quote(arctern_result)
        else:
            update_bool(arctern_result)
        

if __name__ == '__main__':
    
    r = compare_results('/tmp/arctern_results/run_test_st_area.csv', './expected/results/st_area.out')
    # r = compare_results('/tmp/results/test_distance/part-00000-9e90a538-627c-49b6-8fb0-e9f0b263b286-c000.json', './st_distance.out')
    # r = compare_results('/tmp/arctern_results/run_test_st_centroid.json', './expected/results/st_centroid.out')
    # r = compare_results('/tmp/results/test_curvetoline/part-00000-034d8bf0-cc68-4195-8fcf-c23390524865-c000.json', './expected/results/st_curvetoline.out')
    # r = compare_results('/tmp/arctern_results/run_test_st_geometrytype.json', './expected/results/st_geometrytype.out')
    exit(0)

    update_result()
    compare_all()
    
