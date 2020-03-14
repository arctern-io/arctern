drop table if exists t1;
create table t1 (geo1 geometry);
copy t1 from '@path@/transform.csv' DELIMITER '|' csv header;
select updategeometrysrid('t1','geo1',4326);

\o @path@/st_transform.out
select st_transform(geo1,3857) from t1;
\o
