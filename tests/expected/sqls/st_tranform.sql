drop table if exists test_table;
create table test_table (geo geometry);
copy test_table from '@path@/data/transform.csv' DELIMITER '|' csv header;

\o @path@/expected/results/st_transform.out
select st_astext(st_transform(geo,4326),3857)) from test_table;
\o

