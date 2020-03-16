drop table if exists test_envelope;
create table test_envelope (x float, y float);
copy test_envelope from '@path@/data/envelope_aggr2.csv' DELIMITER '|' csv header;

\o @path@/expected/results/st_envelope_aggr2.out
select st_astext(st_envelope(geos)) from (select st_union(st_point(x,y)) as geos from test_envelope) as envelope_aggr;
\o
