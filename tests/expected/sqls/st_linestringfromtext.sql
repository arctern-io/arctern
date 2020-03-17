drop table if exists test_table;
create table test_table (geo text);
copy test_table from '@path@/data/linestringfromtext.csv' DELIMITER '|' csv header;

\o @path@/expected/results/st_linestringfromtext.out
select st_astext(st_linestringfromtext(geo)) from test_table;
\o
