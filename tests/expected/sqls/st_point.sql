drop table if exists t1;
create table t1 (x float,y float);
copy t1 from '@path@/points.csv' DELIMITER '|' csv header;

\o @path@/st_point.out
select st_astext(st_point(x,y)) from t1;
\o

