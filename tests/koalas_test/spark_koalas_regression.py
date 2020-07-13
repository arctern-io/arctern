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
from arctern_spark.geoseries import GeoSeries

ARCTERN_INPUT_DIR = './data/'
ARCTERN_RESULT_DIR = '/tmp/'
EXPECTED_RESULT_DIR = './expected/'

GEO_TYPES = ['POLYGON', 'POINT', 'LINESTRING', 'LINEARRING']
GEO_COLLECTION_TYPES = [
    'MULTIPOLYGON', 'MULTIPOINT', 'MULTILINESTRING', 'GEOMETRYCOLLECTION', 'MULTILINEARRING'
]

EPOCH = 1e-8
EPOCH_FLOAT = 1e-3

unary_func_property_dict = {
    'length': ['length.csv', 'length.out', 'st_length.out'],  # issue 828
    'envelope': ['envelope.csv', 'envelope.out', 'st_envelope.out'],
    'area': ['area.csv', 'area.out', 'st_area.out'],
    'npoints': ['npoints.csv', 'npoints.out', 'st_npoints.out'],
    'is_valid': ['is_valid.csv', 'is_valid.out', 'st_is_valid.out'],
    'centroid': ['centroid.csv', 'centroid.out', 'st_centroid.out'],
    'convex_hull': ['convex_hull.csv', 'convex_hull.out', 'st_convex_hull.out'],
    'exterior': ['exterior.csv', 'exterior.out', 'st_exterior.out'],
    'boundary': ['boundary.csv', 'boundary.out', 'st_boundary.out'],
    'is_empty': ['is_empty.csv', 'is_empty.out', 'st_is_empty.out'],
    'is_simple': ['is_simple.csv', 'is_simple.out', 'st_is_simple.out'],
}

unary_func_dict = {
    'envelope_aggr': ['envelope_aggr.csv', 'envelope_aggr.out', 'st_envelope_aggr.out', None],
    'simplify': ['simplify.csv', 'simplify.out', 'st_simplify.out', [1]],
    'buffer': ['buffer.csv', 'buffer.out', 'st_buffer.out', [1]],
    'unary_union': ['unary_union.csv', 'unary_union.out', 'st_unary_union.out', None],
    'as_geojson': ['as_geojson.csv', 'as_geojson.out', 'st_as_geojson.out', None],
    'precision_reduce': ['precision_reduce.csv', 'precision_reduce.out', 'st_precision_reduce.out', [1]],
    'translate': ['translate.csv', 'translate.out', 'st_translate.out', [2, 2]],
    'affine': ['affine.csv', 'affine.out', 'st_affine.out', [1, 1, 1, 1, 1, 1]],
    'scale': ['scale.csv', 'scale.out', 'st_scale.out', [1, 2, (0, 0)]],
    'rotate': ['rotate.csv', 'rotate.out', 'st_rotate.out', [180, (0, 0)]],
    'to_crs': ['to_crs.csv', 'to_crs.out', 'st_to_crs.out', ['\'EPSG:4326\'']],
    'curve_to_line': ['curve_to_line.csv', 'curve_to_line.out', 'st_curve_to_line.out', None]
}

binary_func_dict = {
    'within': ['within.csv', 'within.out', 'st_within.out'],
    'equals': ['equals.csv', 'equals.out', 'st_equals.out'],
    'distance': ['distance.csv', 'distance.out', 'st_distance.out'],
    'contains': ['contains.csv', 'contains.out', 'st_contains.out'],
    'crosses': ['crosses.csv', 'crosses.out', 'st_crosses.out'],
    'disjoint': ['disjoint.csv', 'disjoint.out', 'st_disjoint.out'],
    'overlaps': ['overlaps.csv', 'overlaps.out', 'st_overlaps.out'],
    'touches': ['touches.csv', 'touches.out', 'st_touches.out'],
    'intersects': ['intersects.csv', 'intersects.out', 'st_intersects.out'],
    'intersection': ['intersection.csv', 'intersection.out', 'st_intersection.out'],
    'symmetric_difference': ['symmetric_difference.csv', 'symmetric_difference.out', 'st_symmetric_difference.out'],
    'hausdorff_distance': ['hausdorff_distance.csv', 'hausdorff_distance.out', 'st_hausdorff_distance.out'],
    'distance_sphere': ['distance_sphere.csv', 'distance_sphere.out', 'st_distance_sphere.out'],
    # 'union':['union.csv','union.out','st_union.out'],  # error
    # 'difference':['difference.csv','difference.out','st_difference.out'],
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


def is_float(str):
    try:
        num = float(str)
        return isinstance(num, float)
    except:
        return False


def convert_str(strr):
    if strr.lower() == 'true' or strr.lower() == 't':
        return True
    if strr.lower() == 'false' or strr.lower() == 'f':
        return False
    if is_float(strr):
        return float(strr)
    return strr


def compare_geometry(geometry_x, geometry_y):
    if geometry_x.upper().endswith('EMPTY') and geometry_y.upper().endswith(
            'EMPTY'):
        return True
    arct = wkt.loads(geometry_x)
    pgis = wkt.loads(geometry_y)
    return arct.equals_exact(pgis, EPOCH) or arct.equals(pgis)


def compare_geometrycollection(geometry_x, geometry_y):
    arct = wkt.loads(geometry_x)
    pgis = wkt.loads(geometry_y)
    return arct.equals_exact(pgis, 1e-10) or arct.equals(pgis)


def compare_floats(geometry_x, geometry_y):
    value_x = float(geometry_x)
    value_y = float(geometry_y)
    if value_x == 0:
        return value_y == 0
    return abs(abs(value_x - value_y) / max(abs(value_x), abs(value_y))) <= EPOCH_FLOAT


def compare_one(result, expect):
    value_x = result[1]
    value_y = expect[1]
    newvalue_x = convert_str(value_x)
    newvalue_y = convert_str(value_y)

    try:
        if newvalue_x == newvalue_y:
            return True

        if isinstance(newvalue_x, bool):
            one_result_flag = (newvalue_x == newvalue_y)
            if not one_result_flag:
                print(result[0], newvalue_x, expect[0], newvalue_y)
            return one_result_flag

        if isinstance(newvalue_x, (int, float)):
            return compare_floats(newvalue_x, newvalue_y)

        if isinstance(newvalue_x, str):
            newvalue_x = newvalue_x.strip().upper()
            newvalue_y = newvalue_y.strip().upper()

            if (is_empty(newvalue_x) and is_empty(newvalue_y)):
                return True

            if is_geometry(newvalue_x) and is_geometry(newvalue_y):
                one_result_flag = compare_geometry(newvalue_x,
                                                   newvalue_y)
                if not one_result_flag:
                    print(result[0], newvalue_x, expect[0], newvalue_y)
                return one_result_flag

            if is_geometrycollection(newvalue_x) and is_geometrycollection(
                    newvalue_y):
                one_result_flag = compare_geometrycollection(
                    newvalue_x, newvalue_y)
                if not one_result_flag:
                    print(result[0], newvalue_x, expect[0], newvalue_y)
                return one_result_flag
            return False

    except ValueError as ex:
        print(repr(ex))
        return False


def compare_results(arctern_results, postgis_results):
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
        print('arctern koalas results count is not consist with expected data.')
        return False

    for arctern_res_item, postgis_res_item in zip(
            arct_arr, pgis_arr):
        res = compare_one(arctern_res_item,
                          postgis_res_item)
        flag = flag and res
    return flag


def compare_all():
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

        res = compare_results(arct_result, pgis_result)
        if res:
            print('Arctern test: %s, result: PASSED' % name[:-4])
        else:
            print('Arctern test: %s, result: FAILED' % name[:-4])

        flag = flag and res
    return flag


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
    input_csv_path = ARCTERN_INPUT_DIR + input_csv
    output_csv_path = ARCTERN_RESULT_DIR + output_csv
    col1, col2 = read_csv2arr(input_csv_path)
    assert len(col1) == len(col2)
    geo_s1 = GeoSeries(col1)
    geo_s2 = GeoSeries(col2)
    if func_name == 'distance_sphere':
        geo_s1.set_crs('EPSG:4326')
        geo_s2.set_crs('EPSG:4326')
    test_codes = 'geo_s1.' + func_name + '(geo_s2)'
    if func_name in ['intersection', 'symmetric_difference']:
        test_codes = test_codes + '.to_wkt()'
    if func_name == 'equals':
        test_codes = 'geo_s1.geom_equals(geo_s2)'
    res = eval(test_codes).sort_index()
    write_arr2csv(output_csv_path, res.tolist())


# This is only for debug
def test_binary_func1(func_name, input_csv, output_csv):
    input_csv_path = ARCTERN_INPUT_DIR + input_csv
    output_csv_path = ARCTERN_RESULT_DIR + output_csv
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
    input_csv_path = ARCTERN_INPUT_DIR + input_csv
    output_csv_path = ARCTERN_RESULT_DIR + output_csv
    col1, col2 = read_csv2arr(input_csv_path)
    assert len(col2) == 0
    geo_s1 = GeoSeries(col1)
    test_codes = 'geo_s1.' + func_name
    if func_name in need_to_wkt_list:
        test_codes += '.to_wkt()'
    res = eval(test_codes).sort_index()
    write_arr2csv(output_csv_path, res.tolist())


def test_unary_func(func_name, input_csv, output_csv, params):
    input_csv_path = ARCTERN_INPUT_DIR + input_csv
    output_csv_path = ARCTERN_RESULT_DIR + output_csv
    col1, col2 = read_csv2arr(input_csv_path)
    assert len(col2) == 0
    geo_s1 = GeoSeries(col1)
    if func_name == 'to_crs':
        geo_s1.set_crs('EPSG:3857')
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

    test_status = compare_all()
    print(test_status)
