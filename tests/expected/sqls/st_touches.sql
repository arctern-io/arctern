drop table if exists t1;
create table t1 (geo1 geometry, geo2 geometry);
copy t1 from '@path@/data/touches.csv' DELIMITER '|' csv header;

\o @path@/expected/results/st_touches.out
select st_touches(geo1, geo2) from t1;
\o

