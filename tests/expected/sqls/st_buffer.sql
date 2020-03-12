drop table if exists test_table;
create table test_table (geo geometry, d float);
copy test_table from '@path@/data/buffer.csv' DELIMITER '|' csv header;

\o @path@/expected/results/st_buffer.out
select st_buffer(geo, d) from test_table;
\o

