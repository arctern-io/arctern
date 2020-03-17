import random
import os
import inspect
import sys
import shutil
import glob
import shapely
from shapely import wkt
from osgeo import ogr
from ogr import *
from util import *

config_file = './config.txt'

geo_types = ['POLYGON', 'POINT', 'LINESTRING']
geo_collection_types = [
    'MULTIPOLYGON', 'MULTIPOINT', 'MULTILINESTRING', 'GEOMETRYCOLLECTION'
]
curve_types = ['CIRCULARSTRING', 'MULTICURVE', 'COMPOUNDCURVE']
surface_types = ['CURVEPOLYGON', 'MULTISURFACE', 'SURFACE']
geo_length_types = ['POINT', 'LINESTRING', 'MULTIPOINT', 'MULTILINESTRING']
geo_area_types = ['POLYGON', 'MULTIPOLYGON']
alist = [
    'run_test_st_area_curve', 'run_test_st_distance_curve',
    'run_test_st_hausdorffdistance_curve'
]
blist = [
    'run_test_st_curvetoline', 'run_test_st_transform',
    'run_test_st_transform1', 'run_test_union_aggr_curve',
    'run_test_st_buffer1', 'run_test_st_buffer_curve',
    'run_test_st_buffer_curve1', 'run_test_st_intersection_curve',
    'run_test_st_simplifypreservetopology_curve'
]


def is_length_types(geo):
    geo = geo.strip().upper()

    for x in geo_length_types:
        if geo.startswith(x) and len(geo) != len(x):
            return True
        else:
            continue

    return False


def is_area_types(geo):
    geo = geo.strip().upper()

    for x in geo_area_types:
        if geo.startswith(x) and len(geo) != len(x):
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
    arr.extend(curve_types)
    arr.extend(surface_types)

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


UNIT = 1e-4
EPOCH = 1e-8
EPOCH_CURVE = 1e-2
EPOCH_SURFACE = 1e-2
# EPOCH_CURVE_RELATIVE = 2e-4
# EPOCH_SURFACE_RELATIVE = 3e-6
# EPOCH_CURVE_RELATIVE = 1e-3
EPOCH_CURVE_RELATIVE = 1e-2
EPOCH_SURFACE_RELATIVE = 1e-1


def compare_length(x, y):
    arct = CreateGeometryFromWkt(x)
    pgis = CreateGeometryFromWkt(y)

    intersection_length = Geometry.Length(Geometry.Intersection(arct, pgis))
    arct_length = Geometry.Length(arct)
    pgis_length = Geometry.Length(pgis)
    print('arctern length: %s, postgis length: %s, intersection length: %s' %
          (str(arct_length), str(pgis_length), str(intersection_length)))
    # result = compare_float(intersection_length, arct_length, pgis_length, EPOCH_CURVE)
    result = compare3float_relative(pgis_length, arct_length,
                                    intersection_length, EPOCH_CURVE_RELATIVE)
    return result


def compare_area(x, y):
    arct = CreateGeometryFromWkt(x)
    pgis = CreateGeometryFromWkt(y)

    intersection_area = Geometry.Area(Geometry.Intersection(arct, pgis))
    arct_area = Geometry.Area(arct)
    pgis_area = Geometry.Area(pgis)
    print('arctern area: %s, postgis area: %s, intersection area: %s' %
          (str(arct_area), str(pgis_area), str(intersection_area)))
    #result = compare_float(intersection_area, arct_area, pgis_area, EPOCH_SURFACE)
    result = compare3float_relative(pgis_area, arct_area, intersection_area,
                                    EPOCH_SURFACE_RELATIVE)
    return result


def compare_geometry(c, x, y):

    if x.upper().endswith('EMPTY') and y.upper().endswith('EMPTY'):
        return True

    if c in blist:
        # if is_length_types(x) and is_length_types(y):
        #     return compare_length(x, y)
        # elif is_area_types(x) and is_area_types(y):
        #     return compare_area(x, y)
        if arc_distance(x, y) < EPOCH_CURVE_RELATIVE:
            return True
        else:
            print('arc distance: %s' % str(arc_distance(x, y)))
            return False
    else:
        arct = wkt.loads(x)
        pgis = wkt.loads(y)
        result = arct.equals_exact(pgis, EPOCH)
        return result


def compare_geometrycollection(c, x, y):

    if c in blist:
        if arc_distance(x, y) < EPOCH_CURVE_RELATIVE:
            return True
        else:
            print('arc distance: %s' % str(arc_distance(x, y)))
            return False
    else:
        arct = wkt.loads(x)
        pgis = wkt.loads(y)
        result = arct.equals(pgis)
        return result


def compare_floats(c, x, y):
    x = float(x)
    y = float(y)
    if x == 0:
        if y == 0:
            return True
        else:
            return False
    if c in alist:
        precision_error = EPOCH_CURVE_RELATIVE
        return (abs((x - y)) <= precision_error)
        # return compare2float_relative(x, y, precision_error)
    else:
        precision_error = EPOCH
    if abs((x - y)) <= precision_error:
        return True
    else:
        # print(x, y)
        return False


def compare_float(x, y, z, precision_error):

    x = float(x)
    y = float(y)
    z = float(z)
    if abs((x - y)) <= precision_error and abs(
        (x - z)) <= precision_error and abs((y - z)) <= precision_error:
        return True
    else:
        # print(x, y)
        return False


def compare2float_relative(x_base, y_check, relative_error):
    x = float(x_base)
    y = float(y_check)
    if ((abs(x_base - y_check)) / (abs(x_base))) <= relative_error:
        return True
    else:
        print('arctern: %s, postgis: %s, precision_error: %s' %
              (str(x), str(y), str(relative_error)))
        return False


def compare3float_relative(x_base, y_check, z_intersection, relative_error):
    return compare2float_relative(x_base, y_check, relative_error) and \
           compare2float_relative(x_base, z_intersection,relative_error) and \
           compare2float_relative(y_check, z_intersection, relative_error)


def compare_curve(x, y):
    arct = CreateGeometryFromWkt(x)
    pgis = CreateGeometryFromWkt(y)

    intersection_length = Geometry.Length(Geometry.Intersection(arct, pgis))
    arct_length = Geometry.Length(arct)
    pgis_length = Geometry.Length(pgis)
    #result = compare_float(intersection_length, arct_length, pgis_length,EPOCH_CURVE)
    result = compare3float_relative(pgis_length, arct_length,
                                    intersection_length, EPOCH_CURVE_RELATIVE)
    return result


def compare_surface(x, y):
    arct = CreateGeometryFromWkt(x)
    pgis = CreateGeometryFromWkt(y)

    intersection_area = Geometry.Area(Geometry.Intersection(arct, pgis))
    arct_area = Geometry.Area(arct)
    pgis_area = Geometry.Area(pgis)

    result = compare3float_relative(pgis_area, arct_area, intersection_area,
                                    EPOCH_SURFACE_RELATIVE)
    #result = compare_float(intersection_area, arct_area, pgis_area, EPOCH_SURFACE)
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


def compare_one(config, result, expect):
    x = result[1]
    y = expect[1]
    c = config
    # print('result: %s' % str(x))
    # print('expect: %s' % str(y))

    x = convert_str(x)
    y = convert_str(y)

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
                flag = compare_geometry(c, x, y)
                if not flag:
                    print(result[0], x, expect[0], y)
                return flag

            elif is_geometrycollection(x) and is_geometrycollection(y):
                flag = compare_geometrycollection(c, x, y)
                if not flag:
                    print(result[0], x, expect[0], y)
                return flag

            # elif is_curve(x) and is_curve(y):
            #     flag = compare_curve(x, y)
            #     if not flag:
            #         print(result[0], x, expect[0], y)
            #     return flag

            # elif is_surface(x) and is_surface(y):
            #     flag = compare_surface(x, y)
            #     if not flag:
            #         print(result[0], x, expect[0], y)
            #     return flag

            else:
                if is_geometrytype(x) and is_geometrytype(y):
                    flag = (x == y)
                    if not flag:
                        print(result[0], x, expect[0], y)
                    return flag

                print(result[0], x, expect[0], y)
                return False

        if isinstance(x, int) or isinstance(x, float):
            flag = compare_floats(c, x, y)
            if not flag:
                print(result[0], x, expect[0], y)
            return flag
    except Exception as e:
        flag = False
    return flag


def compare_results(config, arctern_results, postgis_results):

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
        print(
            'test result size: %s and expected result size: %s, NOT equal, check the two result files'
            % (len(arct_arr), len(pgis_arr)))
        return False

    for x, y in zip(arct_arr, pgis_arr):
        res = compare_one(config, x, y)
        flag = flag and res

    return flag
import random
import os
import inspect
import sys
import shutil           
import glob
import shapely
from shapely import wkt
from osgeo import ogr
from ogr import *
from util import *

# config_file = './config.txt'

geo_types = ['POLYGON', 'POINT', 'LINESTRING']
geo_collection_types = ['MULTIPOLYGON', 'MULTIPOINT', 'MULTILINESTRING', 'GEOMETRYCOLLECTION']
curve_types = ['CIRCULARSTRING','MULTICURVE','COMPOUNDCURVE']
surface_types = ['CURVEPOLYGON','MULTISURFACE','SURFACE']
geo_length_types = ['POINT', 'LINESTRING', 'MULTIPOINT', 'MULTILINESTRING']
geo_area_types = ['POLYGON', 'MULTIPOLYGON']
alist = ['run_test_st_area_curve', 'run_test_st_distance_curve', 'run_test_st_hausdorffdistance_curve']
blist = ['run_test_st_curvetoline', 'run_test_st_transform', 'run_test_st_transform1', 'run_test_union_aggr_curve', 'run_test_st_buffer1', 'run_test_st_buffer_curve', 'run_test_st_buffer_curve1', 'run_test_st_intersection_curve', 'run_test_st_simplifypreservetopology_curve']

def is_length_types(geo):
    geo = geo.strip().upper()

    for x in geo_length_types:
        if geo.startswith(x) and len(geo) != len(x):
            return True
        else:
            continue
    
    return False

def is_area_types(geo):
    geo = geo.strip().upper()

    for x in geo_area_types:
        if geo.startswith(x) and len(geo) != len(x):
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
    arr.extend(curve_types)
    arr.extend(surface_types)

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



UNIT = 1e-4
EPOCH = 1e-8
EPOCH_CURVE = 1e-2
EPOCH_SURFACE = 1e-2
# EPOCH_CURVE_RELATIVE = 2e-4
# EPOCH_SURFACE_RELATIVE = 3e-6
# EPOCH_CURVE_RELATIVE = 1e-3
EPOCH_CURVE_RELATIVE = 1e-2
EPOCH_SURFACE_RELATIVE = 1e-1

def compare_length(x, y):
    arct = CreateGeometryFromWkt(x)
    pgis = CreateGeometryFromWkt(y)

    intersection_length = Geometry.Length(Geometry.Intersection(arct, pgis))
    arct_length = Geometry.Length(arct)
    pgis_length = Geometry.Length(pgis)
    print('arctern length: %s, postgis length: %s, intersection length: %s' % (str(arct_length), str(pgis_length), str(intersection_length)))
    # result = compare_float(intersection_length, arct_length, pgis_length, EPOCH_CURVE)
    result = compare3float_relative(pgis_length, arct_length, intersection_length, EPOCH_CURVE_RELATIVE)
    return result

def compare_area(x, y):
    arct = CreateGeometryFromWkt(x)
    pgis = CreateGeometryFromWkt(y)

    intersection_area = Geometry.Area(Geometry.Intersection(arct, pgis))
    arct_area = Geometry.Area(arct)
    pgis_area = Geometry.Area(pgis)
    print('arctern area: %s, postgis area: %s, intersection area: %s' % (str(arct_area), str(pgis_area), str(intersection_area)))
    #result = compare_float(intersection_area, arct_area, pgis_area, EPOCH_SURFACE)
    result = compare3float_relative(pgis_area, arct_area, intersection_area, EPOCH_SURFACE_RELATIVE)
    return result

def compare_geometry(c, x, y):
    
    if x.upper().endswith('EMPTY') and y.upper().endswith('EMPTY'):
        return True
    
    if c in blist:
        # if is_length_types(x) and is_length_types(y):
        #     return compare_length(x, y)
        # elif is_area_types(x) and is_area_types(y):
        #     return compare_area(x, y)
        if arc_distance(x, y) < EPOCH_CURVE_RELATIVE:
            return True
        else:
            print('arc distance: %s' % str(arc_distance(x, y)))
            return False
    else:
        arct = wkt.loads(x)
        pgis = wkt.loads(y)
        result = arct.equals_exact(pgis, EPOCH)
        return result

def compare_geometrycollection(c, x, y):
    
    if c in blist:
        if arc_distance(x, y) < EPOCH_CURVE_RELATIVE:
            return True
        else:
            print('arc distance: %s' % str(arc_distance(x, y)))
            return False
    else:
        arct = wkt.loads(x)
        pgis = wkt.loads(y)
        result = arct.equals(pgis)
        return result

def compare_floats(c, x, y):
    x = float(x)
    y = float(y)
    if x == 0:
        if y == 0:
            return True
        else:
            return False
    if c in alist:
        precision_error = EPOCH_CURVE_RELATIVE
        return (abs((x - y)) <= precision_error)
        # return compare2float_relative(x, y, precision_error)
    else:
        precision_error = EPOCH
    if abs((x - y)) <= precision_error:
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

def compare2float_relative(x_base, y_check, relative_error):
    x = float(x_base)
    y = float(y_check)
    if ((abs(x_base - y_check)) / (abs(x_base))) <= relative_error:
        return True
    else:
        print('arctern: %s, postgis: %s, precision_error: %s' % (str(x), str(y), str(relative_error)))
        return False

def compare3float_relative(x_base, y_check, z_intersection, relative_error):
    return compare2float_relative(x_base, y_check, relative_error) and \
           compare2float_relative(x_base, z_intersection,relative_error) and \
           compare2float_relative(y_check, z_intersection, relative_error)

def compare_curve(x, y):
    arct = CreateGeometryFromWkt(x)
    pgis = CreateGeometryFromWkt(y)

    intersection_length = Geometry.Length(Geometry.Intersection(arct,pgis))
    arct_length = Geometry.Length(arct)
    pgis_length = Geometry.Length(pgis)
    #result = compare_float(intersection_length, arct_length, pgis_length,EPOCH_CURVE)
    result = compare3float_relative(pgis_length, arct_length, intersection_length,EPOCH_CURVE_RELATIVE)
    return result

def compare_surface(x, y):
    arct = CreateGeometryFromWkt(x)
    pgis = CreateGeometryFromWkt(y)

    intersection_area = Geometry.Area(Geometry.Intersection(arct,pgis))
    arct_area = Geometry.Area(arct)
    pgis_area = Geometry.Area(pgis)

    result = compare3float_relative(pgis_area, arct_area, intersection_area, EPOCH_SURFACE_RELATIVE)
    #result = compare_float(intersection_area, arct_area, pgis_area, EPOCH_SURFACE)
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

def compare_one(config, result, expect):
    x = result[1]
    y = expect[1]
    c = config
    # print('result: %s' % str(x))
    # print('expect: %s' % str(y))

    x = convert_str(x)
    y = convert_str(y)

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
                flag = compare_geometry(c, x, y)
                if not flag:
                    print(result[0], x, expect[0], y)
                return flag

            elif is_geometrycollection(x) and is_geometrycollection(y):
                flag = compare_geometrycollection(c, x, y)
                if not flag:
                    print(result[0], x, expect[0], y)
                return flag

            # elif is_curve(x) and is_curve(y):
            #     flag = compare_curve(x, y)
            #     if not flag:
            #         print(result[0], x, expect[0], y)
            #     return flag

            # elif is_surface(x) and is_surface(y):
            #     flag = compare_surface(x, y)
            #     if not flag:
            #         print(result[0], x, expect[0], y)
            #     return flag

            else:
                if is_geometrytype(x) and is_geometrytype(y):
                    flag = (x == y)
                    if not flag:
                        print(result[0], x, expect[0], y)
                    return flag

                print(result[0], x, expect[0], y)
                return False

        if isinstance(x, int) or isinstance(x, float):
            flag = compare_floats(c, x, y)
            if not flag:
                print(result[0], x, expect[0], y)
            return flag
    except Exception as e:
        flag = False
    return flag


def compare_results(config, arctern_results, postgis_results):

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
        res = compare_one(config, x, y)
        flag = flag and res

    return flag

def parse(config_file):
    with open(config_file, 'r') as f:
        lines = f.readlines()
        xs = [x.strip().split('=') for x in lines if not x.strip().startswith('#')]
    return xs

# # arc_result_dir = './arctern_results'
# arc_result_dir = '/tmp/arctern_results'
# pgis_result_dir = './expected/results'

def compare_all():
    names, table_names, expects = get_tests()

    for name, expect in zip(names, expects):
        
        arct_result = os.path.join(arctern_result, name + '.csv')
        pgis_result = os.path.join(expected_result, expect + '.out')
        print('Arctern test: %s, result compare started, test result: %s, expected result: %s' % (name, arct_result, pgis_result))
        
        if not os.path.isfile(arct_result):
            print('Arctern test: %s, result: FAILED, reason: %s' % (name, 'test result not found [%s]' % arct_result))
            continue

        if not os.path.isfile(pgis_result):
            print('Arctern test: %s, result: FAILED, reason: %s' % (name, 'expected result not found [%s]' % pgis_result))
            continue

        res = compare_results(name, arct_result, pgis_result)
        if res == True:
            print('Arctern test: %s, result: PASSED' % name)
        else:
            print('Arctern test: %s, result: FAILED' % name)


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
    names, table_names, expects = get_tests()

    for x in names:
        arctern_file = os.path.join(arctern_result, x + '.csv')
        
        update_quote(arctern_file)
        update_bool(arctern_file)
        

if __name__ == '__main__':
    update_result()
    # r = compare_results(('run_test_st_intersection_curve', ''), '/tmp/arctern_results/run_test_st_intersection_curve.csv', './expected/results/st_intersection_curve.out')
    # exit(0)

    compare_all()


