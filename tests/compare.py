# Copyright (C) 2019-2020 Zilliz. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#	  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# import glob
from shapely import wkt
from ogr import Geometry
from ogr import CreateGeometryFromWkt
# from util import *


GEO_TYPES = ['POLYGON', 'POINT', 'LINESTRING']
GEO_COLLECTION_TYPES = [
    'MULTIPOLYGON', 'MULTIPOINT', 'MULTILINESTRING', 'GEOMETRYCOLLECTION'
]
CURVE_TYPES = ['CIRCULARSTRING', 'MULTICURVE', 'COMPOUNDCURVE']
SURFACE_TYPES = ['CURVEPOLYGON', 'MULTISURFACE', 'SURFACE']
GEO_LENGTH_TYPES = ['POINT', 'LINESTRING', 'MULTIPOINT', 'MULTILINESTRING']
GEO_AREA_TYPES = ['POLYGON', 'MULTIPOLYGON']
ALIST = [
    'run_test_st_area_curve', 'run_test_st_distance_curve',
    'run_test_st_hausdorffdistance_curve'
]
BLIST = [
    'run_test_st_curvetoline', 'run_test_st_transform',
    'run_test_st_transform1', 'run_test_union_aggr_curve',
    'run_test_st_buffer1', 'run_test_st_buffer_curve',
    'run_test_st_buffer_curve1', 'run_test_st_intersection_curve',
    'run_test_st_simplifypreservetopology_curve'
]


def is_length_types(geo):
    """Determine whether a geometry is point/linestring/multipoint/multilinestring."""
    geo = geo.strip().upper()

    for a_geo_type_in_geo_length_types_list in GEO_LENGTH_TYPES:
        if geo.startswith(a_geo_type_in_geo_length_types_list) and \
                len(geo) != len(a_geo_type_in_geo_length_types_list):
            return True

        continue

    return False


def is_area_types(geo):
    """Determine whether a geometry is polygon/multipolygon."""
    geo = geo.strip().upper()

    for a_geo_type_in_geo_area_types_list in GEO_AREA_TYPES:
        if geo.startswith(a_geo_type_in_geo_area_types_list) and \
                len(geo) != len(a_geo_type_in_geo_area_types_list):
            return True
        # else:
        continue

    return False


def is_geometrytype(geo):
    """Determine whether a given string is to describe a geometry type, like 'point' for example."""
    geo = geo.strip().upper()

    arr = []
    arr.extend(GEO_TYPES)
    arr.extend(GEO_COLLECTION_TYPES)
    arr.extend(CURVE_TYPES)
    arr.extend(SURFACE_TYPES)

    for a_geo_type_in_all_geo_types_list in arr:
        if a_geo_type_in_all_geo_types_list in geo:
            return True
        # else:
        continue

    return False


def is_curve(geo):
    """Determine whether a geometry is curve types, like circularstring/MULTICURVE/COMPOUNDCURVE."""
    geo = geo.strip().upper()

    for a_geo_type_in_curve_geo_types_list in CURVE_TYPES:
        if geo.startswith(a_geo_type_in_curve_geo_types_list):
            return True
        # else:
        continue

    return False


def is_surface(geo):
    """Determine whether a geometry is curve types, like CURVEPOLYGON/MULTISURFACE/SURFACE."""
    geo = geo.strip().upper()

    for a_geo_type_in_surface_geo_types_list in SURFACE_TYPES:
        if geo.startswith(a_geo_type_in_surface_geo_types_list):
            return True
        # else:
        continue

    return False


UNIT = 1e-4
EPOCH = 1e-8
EPOCH_CURVE = 1e-2
EPOCH_SURFACE = 1e-2
EPOCH_CURVE_RELATIVE = 1e-2
EPOCH_SURFACE_RELATIVE = 1e-2


def compare_length(x, y):
    """Compare length of 2 geometry types."""
    arct = CreateGeometryFromWkt(x)
    pgis = CreateGeometryFromWkt(y)

    intersection_length = Geometry.Length(Geometry.Intersection(arct, pgis))
    arct_length = Geometry.Length(arct)
    pgis_length = Geometry.Length(pgis)

    print('arctern length: %s, postgis length: %s, intersection length: %s' %
          (str(arct_length), str(pgis_length), str(intersection_length)))
    # result = compare_float(intersection_length, arct_length, pgis_length, EPOCH_CURVE)
    result = compare3float_relative(
        pgis_length, arct_length, intersection_length, EPOCH_CURVE_RELATIVE)
    return result


def compare_area(x, y):
    """Compare area of 2 geometry types."""
    arct = CreateGeometryFromWkt(x)
    pgis = CreateGeometryFromWkt(y)

    intersection_area = Geometry.Area(Geometry.Intersection(arct, pgis))
    arct_area = Geometry.Area(arct)
    pgis_area = Geometry.Area(pgis)

    print('arctern area: %s, postgis area: %s, intersection area: %s' %
          (str(arct_area), str(pgis_area), str(intersection_area)))
    # result = compare_float(intersection_area, arct_area, pgis_area, EPOCH_SURFACE)
    result = compare3float_relative(
        pgis_area, arct_area, intersection_area, EPOCH_SURFACE_RELATIVE)
    return result


def compare_geometry(c, x, y):
    """Compare whether 2 geometries is 'equal'."""

    if x.upper().endswith('EMPTY') and y.upper().endswith('EMPTY'):
        return True

    if c in BLIST:
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
    """Compare whether 2 geometrycollections is 'equal'."""
    if c in BLIST:
        # print('arc distance: %s' % str(arc_distance(x, y)))
        return arc_distance(x, y) < EPOCH_CURVE_RELATIVE
        #     return True
        # else:
        #     print('arc distance: %s' % str(arc_distance(x, y)))
        #     return False
    # else:

    if not c in BLIST:
        arct = wkt.loads(x)
        pgis = wkt.loads(y)
        result = arct.equals(pgis)
        return result


def compare_floats(c, x, y):
    """Compare whether 2 float values is 'equal'."""
    value_x = float(x)
    value_y = float(y)
    if value_x == 0:
        return value_y == 0

    if c in ALIST:
        precision_error = EPOCH_CURVE_RELATIVE
    else:
        precision_error = EPOCH

    return abs((value_x - value_y)) <= precision_error


def compare_float(x, y, z, precision_error):
    """Compare whether 2 geometries and their intersection is 'equal'."""

    value_x = float(x)
    value_y = float(y)
    value_z = float(z)
    return abs((value_x - value_y)) <= precision_error and \
        abs((value_x - value_z)) <= precision_error and \
        abs((value_y - value_z)) <= precision_error


def compare2float_relative(x_base, y_check, relative_error):
    """Compare whether 2 geometries and their intersection is 'equal', measure with relative."""
    value_x = float(x_base)
    value_y = float(y_check)
    return ((abs(value_x - value_y)) / (abs(value_x))) <= relative_error


def compare3float_relative(x_base, y_check, z_intersection, relative_error):
    """Compare whether 2 geometries and their intersection is 'equal', measure with relative."""
    return compare2float_relative(x_base, y_check, relative_error) and \
        compare2float_relative(x_base, z_intersection, relative_error) and \
        compare2float_relative(y_check, z_intersection, relative_error)


def compare_curve(x, y):
    """Compare whether 2 curve geometries is 'equal'."""
    arct = CreateGeometryFromWkt(x)
    pgis = CreateGeometryFromWkt(y)

    intersection_length = Geometry.Length(Geometry.Intersection(arct, pgis))
    arct_length = Geometry.Length(arct)
    pgis_length = Geometry.Length(pgis)
    # result = compare_float(intersection_length, arct_length, pgis_length,EPOCH_CURVE)
    result = compare3float_relative(
        pgis_length, arct_length, intersection_length, EPOCH_CURVE_RELATIVE)
    return result


def compare_surface(x, y):
    """Compare whether 2 surface geometries is 'equal'."""
    arct = CreateGeometryFromWkt(x)
    pgis = CreateGeometryFromWkt(y)

    intersection_area = Geometry.Area(Geometry.Intersection(arct, pgis))
    arct_area = Geometry.Area(arct)
    pgis_area = Geometry.Area(pgis)

    result = compare3float_relative(
        pgis_area, arct_area, intersection_area, EPOCH_SURFACE_RELATIVE)
    # result = compare_float(intersection_area, arct_area, pgis_area, EPOCH_SURFACE)
    return result


def convert_str(strr):
    """Convert a string to float, if it's not a float value, return string to represent itself."""
    if strr.lower() == 'true' or strr.lower() == 't':
        return True
    elif strr.lower() == 'false' or strr.lower() == 'f':
        return False

    try:
        x = float(strr)
        return x
    except ValueError as e:
        print(repr(e))
        # pass

    return strr


def compare_one(config, result, expect):
    """Compare 1 line of arctern result and expected."""
    value_x = result[1]
    value_y = expect[1]
    # c = config

    newvalue_x = convert_str(value_x)
    newvalue_y = convert_str(value_y)

    try:
        if isinstance(newvalue_x, bool):
            flag = (newvalue_x == newvalue_y)
            if not flag:
                print(result[0], newvalue_x, expect[0], newvalue_y)
            return flag

        if isinstance(newvalue_x, str):
            newvalue_x = newvalue_x.strip().upper()
            newvalue_y = newvalue_y.strip().upper()

            # check order : empty -> GEO_TYPES -> geocollection_types -> curve -> surface
            if (is_empty(newvalue_x) and is_empty(newvalue_y)):
                return True

            elif is_geometry(newvalue_x) and is_geometry(newvalue_y):
                flag = compare_geometry(config, newvalue_x, newvalue_y)
                if not flag:
                    print(result[0], newvalue_x, expect[0], newvalue_y)
                return flag

            elif is_geometrycollection(newvalue_x) and is_geometrycollection(newvalue_y):
                flag = compare_geometrycollection(
                    config, newvalue_x, newvalue_y)
                if not flag:
                    print(result[0], newvalue_x, expect[0], newvalue_y)
                return flag
            else:
                if is_geometrytype(newvalue_x) and is_geometrytype(newvalue_y):
                    flag = (newvalue_x == newvalue_y)
                    if not flag:
                        print(result[0], newvalue_x, expect[0], newvalue_y)
                    return flag

                print(result[0], newvalue_x, expect[0], newvalue_y)
                return False

        if isinstance(newvalue_x, int) or isinstance(newvalue_x, float):
            flag = compare_floats(config, newvalue_x, newvalue_y)
            if not flag:
                print(result[0], newvalue_x, expect[0], newvalue_y)
            return flag
    except ValueError as e:
		print(repr(e))
        flag = False
    return flag


def compare_results(config, arctern_results, postgis_results):
    """Compare the result of arctern function and expected."""
    with open(arctern_results, 'r') as arctern_result_file:
        arct_arr = []
        for (num, value) in enumerate(arctern_result_file, 1):
            if value.strip() != '':
                arct_arr.append((num, value.strip()))

    with open(postgis_results, 'r') as postgis_result_file:
        pgis_arr = []
        for (num, value) in enumerate(postgis_result_file, 1):
            if value.strip() != '':
                pgis_arr.append((num, value.strip()))

    flag = True

    if len(arct_arr) != len(pgis_arr):
        print(
            'test result size: %s and expected result size: %s, NOT equal, check the two result files'
            % (len(arct_arr), len(pgis_arr)))
        return False

    for a_line_in_arctern_result_file, a_line_in_postgis_result_file in zip(arct_arr, pgis_arr):
        res = compare_one(config, a_line_in_arctern_result_file, a_line_in_postgis_result_file)
        flag = flag and res

    return flag


def compare_all():
    """Compare all the results of arctern functions and expected."""
    flag = True
    names, table_names, expects = get_tests()

    for name, expect in zip(names, expects):

        arct_result = os.path.join(arctern_result, name + '.csv')
        pgis_result = os.path.join(expected_result, expect + '.out')
        print('Arctern test: %s, result compare started, test result: %s, expected result: %s' % (
            name, arct_result, pgis_result))

        if not os.path.isfile(arct_result):
            print('Arctern test: %s, result: FAILED, reason: %s' %
                  (name, 'test result not found [%s]' % arct_result))
            continue

        if not os.path.isfile(pgis_result):
            print('Arctern test: %s, result: FAILED, reason: %s' %
                  (name, 'expected result not found [%s]' % pgis_result))
            continue

        res = compare_results(name, arct_result, pgis_result)
        if res == True:
            print('Arctern test: %s, result: PASSED' % name)
        else:
            print('Arctern test: %s, result: FAILED' % name)

        flag = flag and res

    return flag


def update_quote(file_path):
    """Update quotes of the original spark results."""
    with open(file_path, 'r') as the_result_file_from_spark:
        content = the_result_file_from_spark.read()
        update = content.replace(r'"', '')
    with open(file_path, 'w') as the_result_file_from_spark:
        the_result_file_from_spark.write(update)


def update_bool(file_path):
    """Update bool values of the original spark results file."""
    with open(file_path, 'r') as the_result_file_from_spark_for_read_and_abbr_not_allowed_by_pylint:
        content = the_result_file_from_spark_for_read_and_abbr_not_allowed_by_pylint.read()
        update = content.replace('true', 'True').replace('false', 'False')
    with open(file_path, 'w') as the_result_file_from_spark_for_write_and_abbr_not_allowed_by_pylint:
        the_result_file_from_spark_for_write_and_abbr_not_allowed_by_pylint.write(
            update)


def update_result():
    """Update the original spark results."""
    names, table_names, expects = get_tests()

    for name in names:
        arctern_file = os.path.join(arctern_result, name + '.csv')

        update_quote(arctern_file)
        update_bool(arctern_file)


if __name__ == '__main__':
    update_result()

    flag = compare_all()
    if not flag:
        exit(1)
