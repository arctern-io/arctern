drop table if exists t1;
create table t1 (geo1 geometry);
copy t1 from '@path@/simplifypreservetopology.csv' DELIMITER '|' csv header;

\o @path@/st_simplifypreservetopology.out
select st_astext(st_simplifypreservetopology(geo1,1)) from t1;
\o
