drop table if exists t1;
create table t1 (geo1 geometry);
copy t1 from '@path@/data/length.csv' DELIMITER '|' csv header;

\o @path@/expected/results/st_length.out
select st_length(geo1) from t1;
\o

