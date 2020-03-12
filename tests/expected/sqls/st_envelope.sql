drop table if exists t1;
create table t1 (geo1 geometry);
copy t1 from '@path@/data/envelope.csv' DELIMITER '|' csv header;

\o @path@/expected/results/st_envelope.out
select st_astext(st_envelope(geo1)) from t1;
\o

