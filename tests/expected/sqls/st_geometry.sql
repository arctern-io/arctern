drop table if exists t1;
create table t1 (geo1 geometry);
copy t1 from '@path@/geom.csv' DELIMITER '|' csv header;

\o @path@/st_geometry.out
select st_geometrytype(geo1) from t1;
\o

