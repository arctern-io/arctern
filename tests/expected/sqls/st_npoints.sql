drop table if exists t1;
create table t1 (geo1 geometry);
copy t1 from '@path@/data/npoints.csv' DELIMITER '|' csv header;

\o @path@/expected/results/st_npoints.out
select st_npoints(geo1) from t1;
\o

