
basedir=.
datadir=data
sqldir=expected/sqls
csv_file_name=$1
sql_file_name=$2
target_sql=test.sql

scp ${basedir}/${datadir}/${csv_file_name} zc@192.168.2.28:/home/zc/test/liangliu/${datadir}/${csv_file_name}
scp ${basedir}/${sqldir}/${sql_file_name} zc@192.168.2.28:/home/zc/test/liangliu/${target_sql}