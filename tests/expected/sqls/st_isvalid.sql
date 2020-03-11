drop table if exists t1;
create table t1 (geo1 geometry);
copy t1 from '@path@/data/isvalid.csv' DELIMITER '|' csv header;

\o @path@/expected/results/st_isvalid.out
select st_isvalid(geo1) from t1;
\o

