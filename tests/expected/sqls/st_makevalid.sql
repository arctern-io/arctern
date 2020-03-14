drop table if exists t1;
create table t1 (geo1 geometry);
copy t1 from '@path@/makevalid.csv' DELIMITER '|' csv header;

\o @path@/st_makevalid.out
select st_astext(st_makevalid(geo1)) from t1;
\o
