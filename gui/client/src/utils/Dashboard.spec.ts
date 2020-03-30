import {isDashboardReady, getDashboardById} from './Dashboard';

test('isDashboardReady', () => {
  const d1 = {};
  const widgetSettings = {};
  expect(isDashboardReady(d1, widgetSettings)).toBe(false);

  const d2 = {id: 1};
  expect(isDashboardReady(d2, widgetSettings)).toBe(false);

  const d3 = {id: 1, title: '1'};
  expect(isDashboardReady(d3, widgetSettings)).toBe(false);

  const d4 = {id: 1, title: '1', userId: 'zilliz'};
  expect(isDashboardReady(d4, widgetSettings)).toBe(false);

  const d5 = {id: 1, title: '1', userId: 'zilliz', configs: []};
  expect(isDashboardReady(d5, widgetSettings)).toBe(false);

  const d6 = {id: 1, title: '1', userId: 'zilliz', configs: [], sources: ['1']};
  expect(isDashboardReady(d6, widgetSettings)).toBe(true);

  const mockDashboard = `{"id":10,"demo":false,"title":"3d","userId":"infini","configs":[{"type":"LineChart","title":"","source":"nyc_taxi","layout":{"w":13,"h":12,"x":17,"y":0,"i":"id_2xz75a86kh1","minW":3,"minH":1,"moved":false,"static":false},"dimensions":[{"name":"x","format":"auto","type":"DATE","value":"tpep_dropoff_datetime","label":"tpep_dropoff_datetime","isBinned":true,"extract":false,"as":"x","min":"Wed Apr 01 00:01:45 2009","max":"Fri May 01 00:44:00 2009","extent":["Wed Apr 01 00:01:45 2009","Fri May 01 00:44:00 2009"],"staticRange":["Wed Apr 01 00:01:45 2009","Fri May 01 00:44:00 2009"],"timeBin":"day","binningResolution":"1d"},{"name":"color","format":"auto","type":"text","value":"VendorID","label":"VendorID","as":"color"}],"measures":[{"type":"int4","name":"y0","value":"buildingid_pickup","label":"buildingid_pickup","as":"measure_4566bmz5vmh","format":"auto","isCustom":false,"isRecords":false,"expression":"avg"}],"colorItems":[{"colName":"VendorID","value":"VTS","label":"VTS","color":"#ff9f7f","as":"VTS"},{"colName":"VendorID","value":"CMT","label":"CMT","color":"#32C5E9","as":"CMT"},{"colName":"VendorID","value":"DDS","label":"DDS","color":"#67E0E3","as":"DDS"}],"filter":{},"selfFilter":{"selfFilter_color":{"type":"filter","expr":{"type":"in","set":["VTS","CMT","DDS"],"expr":"VendorID"}}},"isShowRange":false,"id":"id_2xz75a86kh1"},{"type":"Scatter3d","title":"","source":"nyc_taxi","layout":{"w":17,"h":12,"x":0,"y":0,"i":"id_4ck2ittjt3m","minW":1,"minH":1,"moved":false,"static":false},"dimensions":[],"measures":[{"type":"float8","name":"x","value":"total_amount","label":"total_amount","as":"x","format":"auto","isCustom":false,"isRecords":false,"expression":"project"},{"type":"float8","name":"y","value":"trip_distance","label":"trip_distance","as":"y","format":"auto","isCustom":false,"isRecords":false,"expression":"project"},{"type":"float8","name":"z","value":"tip_amount","label":"tip_amount","as":"z","format":"auto","isCustom":false,"isRecords":false,"expression":"project"},{"type":"float8","name":"color","value":"tip_amount","label":"tip_amount","as":"color","format":"auto","isCustom":false,"isRecords":false,"expression":"project"}],"colorItems":[],"filter":{},"selfFilter":{},"limit":16001,"colorKey":"blue_green_yellow","ruler":{"min":0,"max":20.385},"rulerBase":{"min":0,"max":100},"id":"id_4ck2ittjt3m"}],"createdAt":"Thu, 02 Jan 2020 07:10:00 GMT","modifyAt":"Thu, 02 Jan 2020 07:10:00 GMT","sources":["car_move","nyc_taxi","wuxian_wifi"],"sourceOptions":{"car_move":[{"colName":"alarm","dataType":"text","type":"text"},{"colName":"altitude","dataType":"double precision","type":"float8"},{"colName":"car_num","dataType":"text","type":"text"},{"colName":"car_speed","dataType":"double precision","type":"float8"},{"colName":"car_type","dataType":"text","type":"text"},{"colName":"colour","dataType":"integer","type":"int4"},{"colName":"direction","dataType":"double precision","type":"float8"},{"colName":"error_type","dataType":"integer","type":"int4"},{"colName":"event","dataType":"integer","type":"int4"},{"colName":"gps_latitude","dataType":"double precision","type":"float8"},{"colName":"gps_longitude","dataType":"double precision","type":"float8"},{"colName":"gps_speed","dataType":"integer","type":"int4"},{"colName":"gps_time","dataType":"text","type":"text"},{"colName":"latitude","dataType":"double precision","type":"float8"},{"colName":"longitude","dataType":"double precision","type":"float8"},{"colName":"mileage","dataType":"bigint","type":"int8"},{"colName":"operation_code","dataType":"bigint","type":"int8"},{"colName":"system_time","dataType":"timestamp without time zone","type":"timestamp"}],"car_moverowCount":500000,"nyc_taxi":[{"colName":"buildingid_dropoff","dataType":"integer","type":"int4"},{"colName":"buildingid_pickup","dataType":"integer","type":"int4"},{"colName":"buildingtext_dropoff","dataType":"text","type":"text"},{"colName":"buildingtext_pickup","dataType":"text","type":"text"},{"colName":"dropoff_latitute","dataType":"double precision","type":"float8"},{"colName":"dropoff_longitute","dataType":"double precision","type":"float8"},{"colName":"fare_amount","dataType":"double precision","type":"float8"},{"colName":"passenger_count","dataType":"integer","type":"int4"},{"colName":"pickup_latitude","dataType":"double precision","type":"float8"},{"colName":"pickup_longitude","dataType":"double precision","type":"float8"},{"colName":"tip_amount","dataType":"double precision","type":"float8"},{"colName":"total_amount","dataType":"double precision","type":"float8"},{"colName":"tpep_dropoff_datetime","dataType":"timestamp without time zone","type":"timestamp"},{"colName":"tpep_pickup_datetime","dataType":"timestamp without time zone","type":"timestamp"},{"colName":"trip_distance","dataType":"double precision","type":"float8"},{"colName":"VendorID","dataType":"text","type":"text"}],"nyc_taxirowCount":500000,"wuxian_wifi":[{"colName":"cap_time","dataType":"timestamp without time zone","type":"timestamp"},{"colName":"channel","dataType":"integer","type":"int4"},{"colName":"infrastructure_mode","dataType":"text","type":"text"},{"colName":"live_speed_bps","dataType":"bigint","type":"int8"},{"colName":"mac","dataType":"text","type":"text"},{"colName":"network_type","dataType":"text","type":"text"},{"colName":"num_id","dataType":"integer","type":"int4"},{"colName":"pos_latitude","dataType":"double precision","type":"float8"},{"colName":"pos_longitude","dataType":"double precision","type":"float8"},{"colName":"privacy","dataType":"text","type":"text"},{"colName":"rates","dataType":"text","type":"text"},{"colName":"rssi","dataType":"integer","type":"int4"},{"colName":"seq_num","dataType":"integer","type":"int4"},{"colName":"ssid","dataType":"text","type":"text"}],"wuxian_wifirowCount":275910}}`;

  localStorage.setItem(`infini.dashboard:10`, mockDashboard);
  expect(JSON.stringify(getDashboardById(10))).toBe(mockDashboard);

  const newDashboard = getDashboardById(100);
  expect(newDashboard.id).toBe(100);
  expect(newDashboard.demo).toBe(false);
  expect(newDashboard.userId).toBe('infini');
  expect(newDashboard.configs).not.toBeUndefined();
  expect(newDashboard.configs.length).toBe(0);
  expect(newDashboard.sources).not.toBeUndefined();
  expect(newDashboard.sources.length).toBe(0);
  expect(newDashboard.sourceOptions).toEqual({});
});
