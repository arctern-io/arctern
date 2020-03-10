drop table if exists test_table;
create table test_table (x_min float, y_min float, x_max float, y_max float);

insert into test_table values
(10.1, 91.9, 19.7, 98.3),
(16.1, 93.3, 16.6, 94.0),
(11.0, 88.3, 18.7, 98.2),
(13.9, 82.2, 19.1, 83.4),
(12.0, 81.5, 16.2, 90.6),
(10.4, 87.5, 11.7, 92.2),
(15.5, 88.7, 18.6, 98.4),
(14.8, 83.0, 16.9, 85.6),
(10.8, 83.9, 16.5, 84.4),
(12.5, 80.8, 14.8, 97.1);   


select st_astext(st_union(myshape)) from (select st_makeenvelope(x_min,y_min,x_max,y_max) as myshape from test_table) as union_aggr;
