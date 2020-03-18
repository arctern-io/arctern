drop table if exists test_table;
create table test_table (geo text);
copy test_table from '@path@/data/polygonfromtext.csv' DELIMITER '|' csv header;

\o @path@/expected/results/st_polygonfromtext.out
select st_astext(st_polygonfromtext(geo)) from test_table;
\o
