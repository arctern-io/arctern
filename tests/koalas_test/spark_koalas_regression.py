# Copyright (C) 2019-2020 Zilliz. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from shapely import wkt

GEO_TYPES = ['POLYGON', 'POINT', 'LINESTRING', 'LINEARRING']
GEO_COLLECTION_TYPES = [
    'MULTIPOLYGON', 'MULTIPOINT', 'MULTILINESTRING', 'GEOMETRYCOLLECTION', 'MULTILINEARRING'
]
CURVE_TYPES = ['CIRCULARSTRING', 'MULTICURVE', 'COMPOUNDCURVE']
SURFACE_TYPES = ['CURVEPOLYGON', 'MULTISURFACE', 'SURFACE']
GEO_LENGTH_TYPES = ['POINT', 'LINESTRING', 'MULTIPOINT', 'MULTILINESTRING']
GEO_AREA_TYPES = ['POLYGON', 'MULTIPOLYGON']

unary_func_property_dict = {
    # 'length':['length.csv', 'length.out','st_length.out'],  # issue 828
    'envelope': ['envelope.csv', 'envelope.out', 'st_envelope.out'],  # empty error!
    'area': ['area.csv', 'area.out', 'st_area.out'],
    'npoints': ['npoints.csv', 'npoints.out', 'st_npoints.out'],
    'is_valid': ['is_valid.csv', 'is_valid.out', 'st_is_valid.out'],
    'centroid': ['centroid.csv', 'centroid.out', 'st_centroid.out'],  # empty error!
    'convex_hull': ['convex_hull.csv', 'convex_hull.out', 'st_convex_hull.out'],
    'exterior': ['exterior.csv', 'exterior.out', 'st_exterior.out'],  # empty error!
    'boundary': ['boundary.csv', 'boundary.out', 'st_boundary.out'],  # e
    'is_empty': ['is_empty.csv', 'is_empty.out', 'st_is_empty.out'],  # e
    # 'is_simple':['is_simple.csv','is_simple.out'], # e
}

unary_func_dict = {
    'envelope_aggr': ['envelope_aggr.csv', 'envelope_aggr.out', 'st_envelope_aggr.out', None],
    'simplify': ['simplify.csv', 'simplify.out', 'st_simplify.out', [1]],
    'buffer': ['buffer.csv', 'buffer.out', 'st_buffer.out', [1]],
    'unary_union': ['unary_union.csv', 'unary_union.out', 'st_unary_union.out', None],
    'as_geojson': ['as_geojson.csv', 'as_geojson.out', 'st_as_geojson.out', None],
    'precision_reduce': ['precision_reduce.csv', 'precision_reduce.out', 'st_precision_reduce.out', [1]],
    'translate':['translate.csv','translate.out','st_translate.out',[2,2]],
    # 'affine':['affine.csv','affine.out','st_affine.out',[1,2,3,4,5,6]],
    # 'scale':['scale.csv','scale.out','st_scale.out',[1,2,(0 0)]],
    # 'rotate':['rotate.csv','rotate.out','st_rotate.out',[180,(0,0)]],
    # 'to_crs':['to_crs.csv','to_crs.out','st_to_crs.out',['\'EPSG:4326\'']],
    # 'curve_to_line':['curve_to_line.csv','curve_to_line.out','st_curve_to_line.out',None],
}

binary_func_dict = {
    'within': ['within.csv', 'within.out', 'st_within.out'],
    'equals': ['equals.csv', 'equals.out', 'st_equals.out'],
    'distance': ['distance.csv', 'distance.out', 'st_distance.out'],
    'contains': ['contains.csv', 'contains.out', 'st_contains.out'],
    'crosses': ['crosses.csv', 'crosses.out', 'st_crosses.out'],
    'intersects': ['intersects.csv', 'intersects.out', 'st_intersects.out'],
    'intersection': ['intersection.csv', 'intersection.out', 'st_intersection.out'],
    # 'symmetric_difference':['symmetric_difference.csv','symmetric_difference.out','st_symmetric_difference.out'],
    # 'hausdorff_distance':['hausdorff_distance.csv','hausdorff_distance.out','st_hausdorff_distance.out'],
    # 'distance_sphere':['distance_sphere.csv','distance_sphere.out','st_distance_sphere.out'] # e
    #
    # 'overlaps':['overlaps.csv','overlaps.out'],  # error
    # 'touches':['touches.csv','touches.out'],  # error
    # 'union':['union.csv','union.out'],  # error
    # 'difference':['difference.csv','difference.out'],
    # 'disjoint':['disjoint.csv','disjoint.out'],
}


def collect_diff_file_list():
    result_file_list = []
    expected_file_list = []
    for key in binary_func_dict.keys():
        result_file_list.append(binary_func_dict[key][1])
        expected_file_list.append(binary_func_dict[key][2])

    for key in unary_func_dict.keys():
        result_file_list.append(unary_func_dict[key][1])
        expected_file_list.append(unary_func_dict[key][2])

    for key in unary_func_property_dict.keys():
        result_file_list.append(unary_func_property_dict[key][1])
        expected_file_list.append(unary_func_property_dict[key][2])

    return result_file_list, expected_file_list


def is_empty(geo):
    geo = geo.strip().upper()
    return geo.endswith('EMPTY')


def is_point(geo):
    geo = geo.strip().upper()
    return geo.startswith('POINT')


def is_linestring(geo):
    geo = geo.strip().upper()
    return geo.startswith('LINESTRING')


def is_polygon(geo):
    geo = geo.strip().upper()
    return geo.startswith('POLYGON')


def is_geometry(geo):
    geo = geo.strip().upper()
    for x in GEO_TYPES:
        if geo.startswith(x) and len(geo) != len(x):
            return True
        continue
    return False


def is_geometrycollection(geo):
    geo = geo.strip().upper()
    for x in GEO_COLLECTION_TYPES:
        if geo.startswith(x):
            return True
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

        continue

    return False


def is_curve(geo):
    """Determine whether a geometry is curve types, like circularstring/MULTICURVE/COMPOUNDCURVE."""
    geo = geo.strip().upper()

    for a_geo_type_in_curve_geo_types_list in CURVE_TYPES:
        if geo.startswith(a_geo_type_in_curve_geo_types_list):
            return True

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


# def compare_length(geometry_x, geometry_y):
#     """Compare length of 2 geometry types."""
#     arct = CreateGeometryFromWkt(geometry_x)
#     pgis = CreateGeometryFromWkt(geometry_y)
#
#     intersection_length = Geometry.Length(Geometry.Intersection(arct, pgis))
#     arct_length = Geometry.Length(arct)
#     pgis_length = Geometry.Length(pgis)
#
#     # print('arctern length: %s, postgis length: %s, intersection length: %s' %
#     #       (str(arct_length), str(pgis_length), str(intersection_length)))
#     # result = compare_float(intersection_length, arct_length, pgis_length, EPOCH_CURVE)
#     result = compare3float_relative(pgis_length, arct_length,
#                                     intersection_length, EPOCH_CURVE_RELATIVE)
#     return result


def compare_area(geometry_x, geometry_y):
    """Compare area of 2 geometry types."""
    arct = CreateGeometryFromWkt(geometry_x)
    pgis = CreateGeometryFromWkt(geometry_y)

    intersection_area = Geometry.Area(Geometry.Intersection(arct, pgis))
    arct_area = Geometry.Area(arct)
    pgis_area = Geometry.Area(pgis)

    # print('arctern area: %s, postgis area: %s, intersection area: %s' %
    #       (str(arct_area), str(pgis_area), str(intersection_area)))
    # result = compare_float(intersection_area, arct_area, pgis_area, EPOCH_SURFACE)
    result = compare3float_relative(pgis_area, arct_area, intersection_area,
                                    EPOCH_SURFACE_RELATIVE)
    return result


def compare_geometry(config, geometry_x, geometry_y):
    """Compare whether 2 geometries is 'equal'."""
    if geometry_x.upper().endswith('EMPTY') and geometry_y.upper().endswith(
            'EMPTY'):
        return True
    arct = wkt.loads(geometry_x)
    pgis = wkt.loads(geometry_y)
    return arct.equals_exact(pgis, EPOCH)


def compare_geometrycollection(config, geometry_x, geometry_y):
    """Compare whether 2 geometrycollections is 'equal'."""

    arct = wkt.loads(geometry_x)
    pgis = wkt.loads(geometry_y)
    return arct.equals_exact(pgis, 1e-10)


def compare_floats(config, geometry_x, geometry_y):
    """Compare whether 2 float values is 'equal'."""
    value_x = float(geometry_x)
    value_y = float(geometry_y)
    if value_x == 0:
        return value_y == 0

    precision_error = EPOCH

    return abs((value_x - value_y)) <= precision_error


def compare_float(geometry_x, geometry_y, geometry_z, precision_error):
    """Compare whether 2 geometries and their intersection is 'equal'."""

    value_x = float(geometry_x)
    value_y = float(geometry_y)
    value_z = float(geometry_z)
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


def compare_curve(geometry_x, geometry_y):
    """Compare whether 2 curve geometries is 'equal'."""
    arct = CreateGeometryFromWkt(geometry_x)
    pgis = CreateGeometryFromWkt(geometry_y)

    intersection_length = Geometry.Length(Geometry.Intersection(arct, pgis))
    arct_length = Geometry.Length(arct)
    pgis_length = Geometry.Length(pgis)
    # result = compare_float(intersection_length, arct_length, pgis_length,EPOCH_CURVE)
    result = compare3float_relative(pgis_length, arct_length,
                                    intersection_length, EPOCH_CURVE_RELATIVE)
    return result


def compare_surface(geometry_x, geometry_y):
    """Compare whether 2 surface geometries is 'equal'."""
    arct = CreateGeometryFromWkt(geometry_x)
    pgis = CreateGeometryFromWkt(geometry_y)

    intersection_area = Geometry.Area(Geometry.Intersection(arct, pgis))
    arct_area = Geometry.Area(arct)
    pgis_area = Geometry.Area(pgis)

    result = compare3float_relative(pgis_area, arct_area, intersection_area,
                                    EPOCH_SURFACE_RELATIVE)
    return result


def is_float(str):
    try:
        num = float(str)
        return isinstance(num, float)
    except:
        return False


def convert_str(strr):
    """Convert a string to float, if it's not a float value, return string to represent itself."""
    if strr.lower() == 'true' or strr.lower() == 't':
        return True
    if strr.lower() == 'false' or strr.lower() == 'f':
        return False

    if is_float(strr):
        return float(strr)

    return strr


# pylint: disable=too-many-return-statements
# pylint: disable=too-many-branches
def compare_one(config, result, expect):
    """Compare 1 line of arctern result and expected."""
    value_x = result[1]
    value_y = expect[1]

    newvalue_x = convert_str(value_x)
    newvalue_y = convert_str(value_y)

    try:
        if isinstance(newvalue_x, bool):
            one_result_flag = (newvalue_x == newvalue_y)
            if not one_result_flag:
                print(result[0], newvalue_x, expect[0], newvalue_y)
            return one_result_flag

        if isinstance(newvalue_x, str):
            newvalue_x = newvalue_x.strip().upper()
            newvalue_y = newvalue_y.strip().upper()

            # check order : empty -> GEO_TYPES -> geocollection_types -> curve -> surface
            if (is_empty(newvalue_x) and is_empty(newvalue_y)):
                return True

            if is_geometry(newvalue_x) and is_geometry(newvalue_y):
                one_result_flag = compare_geometry(config, newvalue_x,
                                                   newvalue_y)
                if not one_result_flag:
                    print(result[0], newvalue_x, expect[0], newvalue_y)
                return one_result_flag

            if is_geometrycollection(newvalue_x) and is_geometrycollection(
                    newvalue_y):
                one_result_flag = compare_geometrycollection(
                    config, newvalue_x, newvalue_y)
                if not one_result_flag:
                    print(result[0], newvalue_x, expect[0], newvalue_y)
                return one_result_flag

            if is_geometrytype(newvalue_x) and is_geometrytype(newvalue_y):
                one_result_flag = (newvalue_x == newvalue_y)
                if not one_result_flag:
                    print(result[0], newvalue_x, expect[0], newvalue_y)
                return one_result_flag

            return False

        if isinstance(newvalue_x, (int, float)):
            return compare_floats(config, newvalue_x, newvalue_y)
            # if not one_result_flag:
            #     print(result[0], newvalue_x, expect[0], newvalue_y)
            # return one_result_flag
    except ValueError as ex:
        print(repr(ex))
        one_result_flag = False
    return one_result_flag


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

    case_result_flag = True

    if len(arct_arr) != len(pgis_arr):
        print('arctern koalas results count is not consist with expected data.')
        return False

    for arctern_res_item, postgis_res_item in zip(
            arct_arr, pgis_arr):
        res = compare_one(config, arctern_res_item,
                          postgis_res_item)
        case_result_flag = case_result_flag and res

    return case_result_flag


def compare_all():
    ARCTERN_RESULT_DIR = './output/'
    EXPECTED_RESULT_DIR = './expected/'
    results, expects = collect_diff_file_list()
    flag = True

    for name, expect in zip(results, expects):

        arct_result = os.path.join(ARCTERN_RESULT_DIR, name)
        pgis_result = os.path.join(EXPECTED_RESULT_DIR, expect)
        print(
            'Arctern test: %s, result compare started, test result: %s, expected result: %s'
            % (name[:-4], arct_result, pgis_result))

        if not os.path.isfile(arct_result):
            print('Arctern test: %s, result: FAILED, reason: %s' %
                  (name[:-4], 'test result not found [%s]' % arct_result))
            continue

        if not os.path.isfile(pgis_result):
            print('Arctern test: %s, result: FAILED, reason: %s' %
                  (name[:-4], 'expected result not found [%s]' % pgis_result))
            continue

        res = compare_results(name[:-4], arct_result, pgis_result)
        if res:
            print('Arctern test: %s, result: PASSED' % name[:-4])
        else:
            print('Arctern test: %s, result: FAILED' % name[:-4])

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
    with open(
            file_path, 'r'
    ) as the_result_file_from_spark_for_read_and_abbr_not_allowed_by_pylint:
        content = the_result_file_from_spark_for_read_and_abbr_not_allowed_by_pylint.read(
        )
        update = content.replace('true', 'True').replace('false', 'False')
    with open(
            file_path,
            'w') as the_result_file_from_spark_for_write_and_abbr_not_allowed:
        the_result_file_from_spark_for_write_and_abbr_not_allowed.write(update)


def update_result():
    """Update the original spark results."""
    results, expects = collect_diff_file_list()
    ARCTERN_RESULT_DIR = './output/'
    EXPECTED_RESULT_DIR = './expected/'
    for f in results:
        arctern_file = os.path.join(ARCTERN_RESULT_DIR, f)

        update_quote(arctern_file)
        update_bool(arctern_file)


#
# import from compare.py ,These codes need to be refactored later.
import pandas as pd
# from osgeo import ogr
from arctern_spark.geoseries import GeoSeries
from databricks.koalas import Series

input_csv_base_dir = './data/'
output_csv_base_dir = './output/'


def read_csv2arr(input_csv_path):
    import re
    arr = []
    col1 = []
    col2 = []
    with open(input_csv_path) as f:
        rows = [line for line in f][1:]  # csv header should be filtered
    for row in rows:
        arr.append(re.split('[|]', row.strip()))
    if len(arr[0]) == 2:
        for items in arr:
            assert len(items) == 2
            col1.append(items[0])
            col2.append(items[1])
    elif len(arr[0]) == 1:
        for items in arr:
            assert len(items) == 1
            col1.append(items[0])
    else:
        raise Exception('Csv file columns length must be 1 or 2.')
    return col1, col2


def write_arr2csv(output_csv_path, output_arr):
    import csv
    with open(output_csv_path, 'w') as f:
        csv_writer = csv.writer(f, delimiter='|', lineterminator='\n')
        for x in output_arr: csv_writer.writerow([x])


def test_binary_func(func_name, input_csv, output_csv):
    input_csv_path = input_csv_base_dir + input_csv
    output_csv_path = output_csv_base_dir + output_csv
    col1, col2 = read_csv2arr(input_csv_path)
    assert len(col1) == len(col2)
    geo_s1 = GeoSeries(col1)
    geo_s2 = GeoSeries(col2)
    test_codes = 'geo_s1.' + func_name + '(geo_s2)'
    if func_name == 'intersection':
        test_codes = test_codes + '.to_wkt()'
    if func_name == 'equals':
        test_codes = 'geo_s1.geom_equals(geo_s2)'
    res = eval(test_codes).sort_index()
    write_arr2csv(output_csv_path, res.tolist())


# This is only for debug
def test_binary_func1(func_name, input_csv, output_csv):
    input_csv_path = input_csv_base_dir + input_csv
    output_csv_path = output_csv_base_dir + output_csv
    col1, col2 = read_csv2arr(input_csv_path)
    assert len(col1) == len(col2)
    for i in range(0, len(col1)):
        geo_s1 = GeoSeries(col1[i])
        geo_s2 = GeoSeries(col2[i])
        test_codes = 'geo_s1.' + func_name + '(geo_s2)'
        res = eval(test_codes)
        print(res)
        print(i)
        print('----\n')
        # write_arr2csv(output_csv_path,res.tolist())


def test_unary_property_func(func_name, input_csv, output_csv):
    need_to_wkt_list = [
        'envelope',
        'centroid',
        'boundary',
        'convex_hull',
        'exterior'
    ]
    input_csv_path = input_csv_base_dir + input_csv
    output_csv_path = output_csv_base_dir + output_csv
    col1, col2 = read_csv2arr(input_csv_path)
    assert len(col2) == 0
    geo_s1 = GeoSeries(col1)
    test_codes = 'geo_s1.' + func_name
    if func_name in need_to_wkt_list:
        test_codes += '.to_wkt()'
    res = eval(test_codes).sort_index()
    write_arr2csv(output_csv_path, res.tolist())


def test_unary_func(func_name, input_csv, output_csv, params):
    input_csv_path = input_csv_base_dir + input_csv
    output_csv_path = output_csv_base_dir + output_csv
    col1, col2 = read_csv2arr(input_csv_path)
    assert len(col2) == 0
    geo_s1 = GeoSeries(col1)
    comma_flag = False
    param_code = ''
    if params == None:
        test_codes = 'geo_s1.' + func_name + '()'
        if not func_name == 'as_geojson':
            test_codes += '.to_wkt()'
    else:
        for param in params:
            if not comma_flag:
                param_code += str(param)
                comma_flag = True
            else:
                param_code += ',' + str(param)
        test_codes = 'geo_s1.' + func_name + '(' + param_code + ').to_wkt()'
    res = eval(test_codes).sort_index()
    write_arr2csv(output_csv_path, res.tolist())


if __name__ == "__main__":
    # test binary_func
    for key, values in binary_func_dict.items():
        test_binary_func(key, values[0], values[1])
    # test unary_func_property
    for key, values in unary_func_property_dict.items():
        test_unary_property_func(key, values[0], values[1])
    # test unary_func
    for key, values in unary_func_dict.items():
        test_unary_func(key, values[0], values[1], values[3])
    # update_result()
    test_status = compare_all()
    # print(test_status)
