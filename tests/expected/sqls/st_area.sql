drop table if exists t1;
create table t1 (geo1 geometry);
copy t1 from '@path@/area.csv' DELIMITER '|' csv header;

\o @path@/area.out
select st_area(geo1) from t1;
\o
