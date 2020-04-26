import os
import numpy
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

# pylint: disable=redefined-outer-name
# pylint: disable=exec-used

def fetch_log_files(directory):
    log_files = []
    files = os.listdir(directory)
    for f in files:
        if os.path.splitext(f)[1] == '.txt':
            log_files.append(f)
    return log_files

def fetch_func_perf(func_list, log_files):
    parser_key1 = 'time:'
    # declare var res_arrs based on func_name
    res = []
    for func_name in func_list:
        exec('res_{}=[]'.format(func_name))
    for func_name in func_list:
        exec('tmp_{}=[]'.format(func_name))

    for log_file in log_files:
        with open(log_dir + log_file, 'r') as f:
            for line in f:
                line = line.split()
                for func_name in func_list:
                    parser_key2 = 'run_test_'+func_name+','
                    if parser_key1 in line and parser_key2 in line:
                        cur_tmp_name = 'tmp_'+func_name
                        tmp_arr = locals()[cur_tmp_name]
                        tmp_arr.append(float(line[-2]))
        for func_name in func_list:
            target_res_arr_name = 'res_'+func_name
            source_tmp_arr_name = 'tmp_'+func_name
            arr = locals()[target_res_arr_name]
            tmp_arr = locals()[source_tmp_arr_name]
            if tmp_arr:
                arr.append(numpy.mean(tmp_arr))
    for func_name in func_list:
        res_arr = 'res_' + func_name
        res.append(locals()[res_arr])
    return res

def perf_stability_alarm(func_name, res_func_arr):
    warning_str = ' [Warning] : The performance of ' + str(func_name) + ' fluctuates greatly!'

    # warning based on standard deviation
    std_deviation = numpy.std(res_func_arr, ddof=1)
    # print("Performance test standard deviation of %s: "%func_name,end='')
    # print(std_deviation)
    if std_deviation > std_threshold:
        print(warning_str)
    # print perf_test result
      # print('perf_test result for ' + str(func_name) + ' :')
      # print(res_func_arr)
        return 1
    return 0

def plot_perf_regression(func_name, res_func_arr):
    perf_picture = plot_dir + str(func_name) + '_perf.png'
    perf_fig_title = str(func_name) + ' Performance Fig'

    # plot
    index1 = list(range(1, len(res_func_arr) + 1))
    index2 = list(range(1, len(res_func_arr) + 1))
    res_mock = numpy.random.rand(len(index2)) * 10 + numpy.mean(res_func_arr)
    plt.figure(figsize=(16, 4))  # picture size
    # plt.plot(index,res_func_arr,color="b--",label="$input_1k$",linewidth=1)
    plt.plot(index1, res_func_arr, label="$10k$", color="blue", linewidth=1)
    plt.plot(index2, res_mock, label="$110k(Simulated-contrast-test)$", color="red", linestyle='--', linewidth=1)
    plt.xlabel("Version ")  # X label
    plt.ylabel("Cost /ms")  # Y label
    plt.title(perf_fig_title)
    x_major_locator = MultipleLocator(1)
    y_major_locator = MultipleLocator(10)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)
    plt.xlim(0, len(res_func_arr) + 2)
    plt.ylim(min(res_func_arr) - 10, max(res_func_arr) + 10)
    plt.legend()
    # plt.show()
    plt.savefig(perf_picture)
    plt.close('all')
    # print('plot %s performance regression picture success.'%func_name)


func_list_arr = ['st_geomfromgeojson',
'st_geomfromgeojson2',
'st_curvetoline',
'st_point',
'st_isvalid_1',
'st_isvalid_curve',
'st_intersection',
'st_intersection_curve',
'st_convexhull',
'st_convexhull_curve',
'st_buffer',
'st_buffer_curve',
'st_buffer_curve1',
'st_envelope',
'st_envelope_curve',
'st_centroid',
'st_centroid_curve',
'st_length',
'st_length_curve',
'st_area',
'st_area_curve',
'st_distance',
'st_distance_curve',
'st_issimple',
'st_issimple_curve',
'st_npoints',
'st_geometrytype',
'st_geometrytype_curve',
'st_intersects',
'st_intersects_curve',
'st_contains',
'st_contains_curve',
'st_within',
'st_within_curve',
'st_equals_1',
'st_equals_2',
'st_crosses',
'st_crosses_curve',
'st_overlaps',
'st_overlaps_curve',
'st_touches',
'st_touches_curve',
'st_makevalid',
'st_precisionreduce',
'st_polygonfromenvelope',
'st_simplifypreservetopology',
'st_simplifypreservetopology_curve',
'st_hausdorffdistance',
'st_hausdorffdistance_curve',
'st_pointfromtext',
'st_polygonfromtext',
'st_linestringfromtext',
'st_geomfromtext',
'st_geomfromwkt',
'st_astext',
'st_buffer1',
'st_buffer2',
'st_buffer3',
'st_buffer4',
'st_buffer5',
'st_buffer6',
'envelope_aggr_1',
'envelope_aggr_curve',
'envelope_aggr_2',
'union_aggr_2',
'union_aggr_curve',
'st_transform',
'st_transform1',
'none']
log_dir = 'perf/log/'
plot_dir = 'perf/picture/'
# Performance regression standard deviation accuracy tolerance
std_threshold = 3.0

# main invocation
if __name__ == "__main__":
    # res_set is a list that contains historical performance data for all gis functions
    res_set = fetch_func_perf(func_list_arr, fetch_log_files(log_dir))
    assert len(func_list_arr) == len(res_set)

    # produce specific result variable in main for every functions in func_list_arr
    for i, func_name in enumerate(func_list_arr):
        exec('res_{}={}'.format(func_name, res_set[i]))
    # for func_name in func_list_arr:
      # res_arr = 'res_' +  func_name
      # print('------ %s performance history data ----' % func_name)
      # print(locals()[res_arr])

    alarm_num = 0
    plot_num = 0
    plot_failed_func = []
    alarm_func = []
    for func_name in func_list_arr:
        res_arr = 'res_' + func_name
        # plot
        cur_res_arr = locals()[res_arr]
        if cur_res_arr:
            plot_perf_regression(func_name, cur_res_arr)
            plot_num = plot_num + 1
            # performance test stability
            if perf_stability_alarm(func_name, locals()[res_arr]):
                alarm_num = alarm_num + 1
                alarm_func.append(func_name)
        else:
            plot_failed_func.append(func_name)


    print('Plot %s performance regression Fig.'%plot_num)
    print('Plot failed functions list :%s '%plot_failed_func)
    print('There are %s functions alarming!'%alarm_num)
    print('Alarm functions list :%s '%alarm_func)
    # specific funciton test examples
    #plot_perf_regression('st_within',res_st_within)
    #perf_stability_alarm('st_within',res_st_within)
