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


def get_sqls_from_data(function_name, file_path):
    sql_arr = []
    with open(file_path, 'r') as f:
        lines = f.readlines()[1:]
        lines = [x.strip().split('|') for x in lines]
        for line in lines:
            if len(line) == 1:
                if function_name in convexhull:
                    sql = sql_template_5 % (function_name, line[0])
                elif function_name in transform:
                    sql = sql_template_6 % (function_name, line[0])
                elif function_name in simplifypreservetopology:
                    sql = sql_template_7 % (function_name, line[0])
                elif function_name in st_buffer:
                    sql = sql_template_3 % (function_name, line[0])
                else:
                    sql = sql_template_1 % (function_name, line[0])

            if len(line) == 2:
                if function_name in intersection:
                    sql = sql_template_4 % (function_name, line[0], line[1])
                else:
                    sql = sql_template_2 % (function_name, line[0], line[1])

            sql_arr.append(sql)
    return sql_arr


def execute_sql(sql):
    try:
        cur.execute(sql)
        rows = [row for row in cur.fetchall()]
        for r in rows:
            return r[0]
    except Exception as e:
        print(sql)
        # print('sql failed')
        pass


def get_postgis_result(sqls, result_path):
    results = [execute_sql(sql) for sql in sqls]
    with open(result_path, 'w') as f:
        for r in results:
            f.writelines(str(r) + '\n')

function_name = sys.argv[1]
file_path = sys.argv[2]
result_path = sys.argv[3]

if __name__ == '__main__':
    sqls = get_sqls_from_data(function_name, file_path)
    get_postgis_result(sqls, result_path)
