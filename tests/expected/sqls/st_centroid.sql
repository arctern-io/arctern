drop table if exists t1;
create table t1 (geo1 geometry);
copy t1 from '@path@/centroid.csv' DELIMITER '|' csv header;

\o @path@/centroid.out
select st_astext(st_centroid(geo1)) from t1;
\o
