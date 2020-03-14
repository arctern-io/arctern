drop table if exists t1;
create table t1 (geo1 geometry, geo2 geometry);
copy t1 from '@path@/data/distance.csv' DELIMITER '|' csv header;

\o @path@/expected/results/st_distance.out
select st_hausdorffdistance(geo1, geo2) from t1;
\o
