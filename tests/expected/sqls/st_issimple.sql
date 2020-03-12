drop table if exists t1;
create table t1 (geo1 geometry);
copy t1 from '@path@/area.csv' DELIMITER '|' csv header;

\o @path@/st_issimple.out
select st_issimple(geo1) from t1;
\o
