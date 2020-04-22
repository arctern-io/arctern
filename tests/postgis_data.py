import sys
import psycopg2

conn_config = "dbname='postgres' host='192.168.2.36' port=5432 user='postgres'"
conn = psycopg2.connect(conn_config)

cur = conn.cursor()
sql = r"select datname from pg_database;"

sql_template_1 = "select %s('%s'::geometry)"
sql_template_2 = "select %s('%s'::geometry, '%s'::geometry)"
sql_template_3 = "select st_astext(%s('%s'::geometry, 1, 30))"
sql_template_4 = "select st_astext(%s('%s'::geometry, '%s'::geometry))"
sql_template_5 = "select st_astext(%s('%s'::geometry))"
sql_template_6 = "select st_astext(%s(st_geomfromtext('%s',3857),4326))"
sql_template_7 = "select st_astext(%s('%s'::geometry, 1))"


st_buffer = ['st_buffer']
intersection = ['st_intersection']
convexhull = ['st_convexhull', 'st_envelope', 'st_union', 'st_curvetoline', 'st_centroid']
transform = ['st_transform']
simplifypreservetopology = ['st_simplifypreservetopology']

def get_sqls_from_data(function_name, path):
    sql_arr = []
    with open(path, 'r') as f:
        lines = f.readlines()[1:]
        lines = [x.strip().split('|') for x in lines]
        for line in lines:
            if len(line) == 1:
                if function_name in convexhull:
                    sqlstr = sql_template_5 % (function_name, line[0])
                elif function_name in transform:
                    sqlstr = sql_template_6 % (function_name, line[0])
                elif function_name in simplifypreservetopology:
                    sqlstr = sql_template_7 % (function_name, line[0])
                elif function_name in st_buffer:
                    sqlstr = sql_template_3 % (function_name, line[0])
                else:
                    sqlstr = sql_template_1 % (function_name, line[0])

            if len(line) == 2:
                if function_name in intersection:
                    sqlstr = sql_template_4 % (function_name, line[0], line[1])
                else:
                    sqlstr = sql_template_2 % (function_name, line[0], line[1])

            sql_arr.append(sqlstr)
    return sql_arr


def execute_sql(sqlstr):
    try:
        cur.execute(sqlstr)
        rows = cur.fetchall()
        for r in rows:
            return r[0]
    except Exception as e: # pylint: disable=broad-except
        # print(sqlstr)
        # print('sqlstr failed')
        print(str(e))


def get_postgis_result(sqls, path):
    results = [execute_sql(sql) for sql in sqls]
    with open(path, 'w') as f:
        for r in results:
            f.writelines(str(r) + '\n')


if __name__ == '__main__':
    func_name = sys.argv[1]
    file_path = sys.argv[2]
    result_path = sys.argv[3]
    # print(function_name)
    # print(file_path)
    # print(result_path)
    sqlstrs = get_sqls_from_data(func_name, file_path)
    get_postgis_result(sqlstrs, result_path)
