drop table if exists test_table;
create table test_table (geo text);
copy test_table from '@path@/data/pointfromtext.csv' DELIMITER '|' csv header;

\o @path@/expected/results/st_pointfromtext.out
select st_astext(st_pointfromtext(geo)) from test_table;
\o
