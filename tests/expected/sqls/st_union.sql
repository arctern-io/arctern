drop table if exists t1;
create table t1 (geo1 geometry);
copy t1 from '@path@/data/union_aggr.csv' DELIMITER '|' csv header;

\o @path@/expected/results/st_union.out
select st_astext(st_union(geo1)) from t1;
\o

