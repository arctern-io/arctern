drop table if exists t1;
create table t1 (geo1 geometry, geo2 geometry);
copy t1 from '@path@/crosses.csv' DELIMITER '|' csv header;

\o @path@/st_crosses.out
select st_crosses(geo1, geo2) from t1;
\o
