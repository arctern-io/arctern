import random
import os
import inspect
import sys
import shutil           
import glob
# import pygeos
import shapely
from shapely import wkt
from osgeo import ogr
from ogr import *

config_file = './config.txt'

geo_types = ['POLYGON', 'POINT', 'LINESTRING']
geo_collection_types = ['MULTIPOLYGON', 'MULTIPOINT', 'MULTILINESTRING', 'GEOMETRYCOLLECTION']
curve_types = ['CIRCULARSTRING','MULTICURVE','COMPOUNDCURVE']
surface_types = ['CURVEPOLYGON','MULTISURFACE','SURFACE']

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

def is_empty(geo):
    geo = geo.strip().upper()
    if geo.endswith('EMPTY'):
        return True
    else:
        return False


def is_curve(geo):
    geo = geo.strip().upper()

    for x in curve_types:
        if geo.startswith(x):
            return True
        else:
            continue

    return False

def is_surface(geo):
    geo = geo.strip().upper()

    for x in surface_types:
        if geo.startswith(x):
            return True
        else:
            continue

    return False



UNIT = 0.0001
EPOCH = 1e-8
EPOCH_CURVE = 1e-2
EPOCH_SURFACE = 1e-2

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

def compare_float(x, y, z, precision_error):

    x = float(x)
    y = float(y)
    z = float(z)
    if abs((x - y)) <= precision_error and abs((x-z)) <= precision_error and abs((y-z)) <= precision_error:
        return True
    else:
        # print(x, y)
        return False

def compare_curve(x, y):
    arct = CreateGeometryFromWkt(x)
    pgis = CreateGeometryFromWkt(y)

    intersection_length = Geometry.Length(Geometry.Intersection(arct,pgis))
    arct_length = Geometry.Length(arct)
    pgis_length = Geometry.Length(pgis)
    result = compare_float(intersection_length, arct_length, pgis_length,EPOCH_CURVE)
    return result

def compare_surface(x, y):
    arct = CreateGeometryFromWkt(x)
    pgis = CreateGeometryFromWkt(y)

    intersection_area = Geometry.Area(Geometry.Intersection(arct,pgis))
    arct_area = Geometry.Area(arct)
    pgis_area = Geometry.Area(pgis)

    print(intersection_area)
    print(arct_area)
    print(pgis_area)
    result = compare_float(intersection_area, arct_area, pgis_area, EPOCH_SURFACE)
    return result

def convert_str(strr):
    if strr.lower() == 'true' or strr.lower() == 't':
        return True
    elif strr.lower() == 'false' or strr.lower() == 'f':
        return False

    try:
        x = float(strr)
        return x
    except:
        pass

    return strr

def compare_one(result, expect):
    x = result[1]
    y = expect[1]
    # print('result: %s' % str(x))
    # print('expect: %s' % str(y))

    x = convert_str(x)
    y = convert_str(y)

    # if y.strip() == 't':
    #     y = True
    # elif y.strip() == 'f':
    #     y = False

    try:
        if isinstance(x, bool):
            flag = (x == y)
            if not flag:
                print(result[0], x, expect[0], y)
            return flag

        if isinstance(x, str):
            x = x.strip().upper()
            y = y.strip().upper()
        # check order : empty -> geo_types -> geocollection_types -> curve -> surface
            if (is_empty(x) and is_empty(y)):
                return True

            elif is_geometry(x) and is_geometry(y):
                flag = compare_geometry(x, y)
                if not flag:
                    print(result[0], x, expect[0], y)
                return flag

            elif is_geometrycollection(x) and is_geometrycollection(y):
                flag = compare_geometrycollection(x, y)
                if not flag:
                    print(result[0], x, expect[0], y)
                return flag

            elif is_curve(x) and is_curve(y):
                flag = compare_curve(x, y)
                if not flag:
                    print(result[0], x, expect[0], y)
                return flag

            elif is_surface(x) and is_surface(y):
                flag = compare_surface(x, y)
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
#compare.py unittest cases (expected no AssertionError)    
    #test compare EMPTY
    geo1 = 'POINT EMPTY'
    geo2 = 'POINT EMPTY'
    geo3 = 'CIRCULARSTRING EMPTY'
    geo4 = 'POLYGON((0 0,1000000 0,1000000 2000000,0 0))'
    assert True == compare_one([1,geo1],[1,geo2])
    assert True == compare_one([2,geo1],[2,geo3])
    assert False == compare_one([3,geo1],[3,geo4])

    #test geo_types
    geo1 = 'POLYGON((0 0,100000000 0,100000000 100000000,0 0))'
    geo2 = 'POLYGON((0 0,100000000 0,100000000 100000000.000000001,0 0))'
    geo3 = 'POLYGON((0 0,100000000 0,100000000 200000000,0 0))'
    assert True == compare_one([4,geo1],[4,geo2])
    assert False == compare_one([5,geo1],[5,geo3])


    #test geo_collection_types
    geo1 = 'GEOMETRYCOLLECTION (POINT (2 1),LINESTRING (0 0,1 1,2 3),POLYGON((0 0,1000000 0,1000000 1000000,0 0)))'
    geo2 = 'GEOMETRYCOLLECTION (LINESTRING (0 0,1 1,2 3),POLYGON((0 0,1000000.000000000003 0,1000000 1000000,0 0)),POINT(2 1))'
    geo3 = 'GEOMETRYCOLLECTION (POINT (2 1),LINESTRING (0 0,1 2,2 3),POLYGON((0 0,2000000 0,1000000 1000000,0 0)))'
    
    assert True == compare_one([6,geo1],[6,geo2])
    assert False == compare_one([7,geo1],[7,geo3])

    #test curve
    geo1 = 'CIRCULARSTRING (0 2, -1 1,0 0, 0.5 0, 1 0, 2 1, 1 2, 0.5 2, 0 2)'
    geo2 = 'CIRCULARSTRING (0 2, -1 1,0 0, 0.5 0, 1 0, 2 1, 1 2, 0.5 2, 0 2)'
    geo3 = 'CIRCULARSTRING (28 8882, -1 1,0 0, 331.5 0, 1 0, 2 1, 1 2, 0.5 2, 0 2)'
    #geo3 = 'CIRCULARSTRING (0 2, -1 1,0 0, 331.5 0, 1 0, 2 1, 1 2, 0.5 2, 0 2)' # hit assert ex!

    geo4 = 'COMPOUNDCURVE(CIRCULARSTRING(0 2, -1 1,1 0),CIRCULARSTRING( 1 0, 2 1, 1 2),(1 2, 0.5 2, 0 2))'
    geo5 = 'COMPOUNDCURVE(CIRCULARSTRING(0 2, -1 1,1 0),CIRCULARSTRING( 1 0, 2 1, 1 2),(1 2, 0.5 2, 0 2))'
    geo6 = 'COMPOUNDCURVE(CIRCULARSTRING(0 2, -1 1,11 0),CIRCULARSTRING( 11 0, 2 1, 1 2),(1 2, 0.5 2, 0 2))'
    
    geo7 = 'MULTICURVE ((5 5, 3 5, 3 3, 0 3), CIRCULARSTRING (0 0, 0.2 1, 0.5 1.4), COMPOUNDCURVE(LINESTRING(0 2, -1 1,1 0),CIRCULARSTRING( 1 0, 2 1, 1 2),(1 2, 0.5 2, 0 2)))'
    geo8 = 'MULTICURVE ((5 5, 3 5, 3 3, 0 3), CIRCULARSTRING (0 0, 0.2 1, 0.5 1.4), COMPOUNDCURVE(LINESTRING(0 2, -1 1,1 0),CIRCULARSTRING( 1 0, 2 1, 1 2),(1 2, 0.5 2, 0 2)))'
    geo9 = 'MULTICURVE ((5 5, 3 5, 3 3, 0 3), CIRCULARSTRING (0 0, 0.2 1, 0.5 1.4), COMPOUNDCURVE(LINESTRING(0 2, -1 1,1 0),CIRCULARSTRING( 1 0, 2 1, 1 2),(1 2, 0.5 2, 0 3)))'
    
    assert True == compare_one([8,geo1],[8,geo2])
    assert False == compare_one([9,geo1],[9,geo3])
    assert True == compare_one([10,geo4],[10,geo5])
    assert False == compare_one([11,geo4],[11,geo6])
    assert True == compare_one([12,geo7],[12,geo8])
    assert False == compare_one([13,geo7],[13,geo9])

    #test surface 
    geo1 = 'CURVEPOLYGON(CIRCULARSTRING(0 0, 4 0, 4 4, 0 4, 0 0),(1 1, 3 3, 3 1, 1 1))'
    geo2 = 'CURVEPOLYGON(CIRCULARSTRING(0 0, 4 0, 4 4, 0 4, 0 0),(1 1, 3 3, 3 1, 1 1))'
    geo3 = 'CURVEPOLYGON(CIRCULARSTRING(0 0, 4 0, 4 4, 0 4, 0 0),(1 1, 3 3, 3.3 1, 1 1))'

    geo4 = 'MULTISURFACE (CURVEPOLYGON (CIRCULARSTRING (-2 0, -1 -1, 0 0, 1 -1, 2 0, 0 2, -2 0), (-1 0, 0 0.5, 1 0, 0 1, -1 0)), ((7 8, 10 10, 6 14, 4 11, 7 8)))'
    geo5 = 'MULTISURFACE (CURVEPOLYGON (CIRCULARSTRING (-2 0, -1 -1, 0 0, 1 -1, 2 0, 0 2, -2 0), (-1 0, 0 0.5, 1 0, 0 1, -1 0)), ((7 8, 10 10, 6 14, 4 11, 7 8)))'
    geo6 = 'MULTISURFACE (CURVEPOLYGON (CIRCULARSTRING (-2 0, -1 -1, 0 0, 1 -1, 2 0, 0 2, -2 0), (-1 0, 0 0.5, 1 0, 0 1, -1 0)), ((7 8, 10 10, 6 14, 4 13, 7 8)))'
    
    assert True == compare_one([14,geo1],[14,geo2])
    assert False == compare_one([15,geo1],[15,geo3])
    assert True == compare_one([16,geo4],[16,geo5])
    assert False == compare_one([17,geo4],[17,geo6])

# arctern regression test demo code
    #update_result()
    #compare_all()
    
