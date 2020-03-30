import {
  isReadyToRender,
  isRecordExist,
  getValidColumns,
  calStatus,
  measureUsePopUp,
  filterColumns,
  getDefaultTitle,
} from './EditorHelper';
import {PointConfigLackLonMeasure, BarConfig, PieConfigLackDimension} from './SpecHelper';
import PointSetting from '../widgets/PointMap/settings';
import BarSetting from '../widgets/BarChart/settings';
import PieSetting from '../widgets/PieChart/settings';
const columns = [
  {
    colName: 'buildingid_dropoff',
    dataType: 'integer',
    type: 'int4',
  },
  {
    colName: 'buildingid_pickup',
    dataType: 'integer',
    type: 'int4',
  },
  {
    colName: 'buildingtext_dropoff',
    dataType: 'text',
    type: 'text',
  },
  {
    colName: 'buildingtext_pickup',
    dataType: 'text',
    type: 'text',
  },
  {
    colName: 'dropoff_latitute',
    dataType: 'double precision',
    type: 'float8',
  },
  {
    colName: 'dropoff_longitute',
    dataType: 'double precision',
    type: 'float8',
  },
  {
    colName: 'fare_amount',
    dataType: 'double precision',
    type: 'float8',
  },
  {
    colName: 'passenger_count',
    dataType: 'integer',
    type: 'int4',
  },
  {
    colName: 'pickup_latitude',
    dataType: 'double precision',
    type: 'float8',
  },
  {
    colName: 'pickup_longitude',
    dataType: 'double precision',
    type: 'float8',
  },
  {
    colName: 'tip_amount',
    dataType: 'double precision',
    type: 'float8',
  },
  {
    colName: 'total_amount',
    dataType: 'double precision',
    type: 'float8',
  },
  {
    colName: 'tpep_dropoff_datetime',
    dataType: 'timestamp without time zone',
    type: 'timestamp',
  },
  {
    colName: 'tpep_pickup_datetime',
    dataType: 'timestamp without time zone',
    type: 'timestamp',
  },
  {
    colName: 'trip_distance',
    dataType: 'double precision',
    type: 'float8',
  },
  {
    colName: 'VendorID',
    dataType: 'text',
    type: 'text',
  },
];
test('isReadyToRender', () => {
  const PointReady = isReadyToRender(PointConfigLackLonMeasure, PointSetting);
  const BarReady = isReadyToRender(BarConfig, BarSetting);
  const PieReady = isReadyToRender(PieConfigLackDimension, PieSetting);

  expect(PointReady).toStrictEqual({
    sourceReady: {isReady: true},
    dimensionsReady: {isReady: true, lacks: []},
    measuresReady: {
      isReady: false,
      lacks: [
        {
          columnTypes: ['number'],
          expressions: ['gis_mapping_lon'],
          key: 'lon',
          short: 'longtitude',
          type: 'required',
        },
      ],
    },
  });
  expect(BarReady).toStrictEqual({
    sourceReady: {isReady: true},
    dimensionsReady: {isReady: true, lacks: []},
    measuresReady: {isReady: true, lacks: []},
  });
  expect(PieReady).toStrictEqual({
    sourceReady: {isReady: true},
    dimensionsReady: {isReady: false, lacks: [{key: '', short: '', type: 'requiedOneAtLeast'}]},
    measuresReady: {isReady: true, lacks: []},
  });
});
test('isRecordExist', () => {
  expect(isRecordExist(PointConfigLackLonMeasure)).toBe(false);
  expect(isRecordExist(BarConfig)).toBe(false);
  expect(isRecordExist(PieConfigLackDimension)).toBe(false);
});
test('getValidColumns', () => {
  const numRes = getValidColumns(columns, ['number']);
  const dateRes = getValidColumns(columns, ['date']);
  const textRes = getValidColumns(columns, ['text']);
  const numDateRes = getValidColumns(columns, ['number', 'date']);
  const invalidRes = getValidColumns(columns, ['lalallallala']);
  const allRes = getValidColumns(columns, ['number', 'date', 'text']);

  expect(numRes.length).toBe(11);
  expect(dateRes.length).toBe(2);
  expect(textRes.length).toBe(3);
  expect(numDateRes.length).toBe(13);
  expect(invalidRes.length).toBe(0);
  expect(allRes.length).toBe(16);
});
test('calStatus', () => {
  const textDimension = {type: 'text'};
  const numBinDimension = {isNotUseBin: true, value: 'ahhhh'};
  const dateBinDimension = {type: 'timestamp', value: 'ahhhh'};
  const textMeasure = {value: 'ahhhh', expression: 'unique', type: 'text'};
  const numMeasure = {expression: 'avg', value: 'ahhhh'};
  const staticMeasure = {expression: 'lalallalal', value: 'ahhhh'};

  const dRes0 = calStatus(textDimension, {}, 'dimension');
  const dRes1 = calStatus(numBinDimension, {isNotUseBin: true}, 'dimension');
  const dRes2 = calStatus({value: 'ahhhh'}, {}, 'dimension');
  const dRes3 = calStatus(dateBinDimension, {isNotUseBin: true}, 'dimension');
  const dRes4 = calStatus(dateBinDimension, {}, 'dimension');

  const mRes0 = calStatus(textMeasure, {expressions: ['avg', 'min', 'unique']}, 'measure');
  const mRes1 = calStatus(textMeasure, {expressions: ['min']}, 'measure');
  const mRes2 = calStatus(numMeasure, {expressions: ['avg']}, 'measure');
  const mRes3 = calStatus(numMeasure, {expressions: ['avg', 'min']}, 'measure');
  const mRes4 = calStatus(staticMeasure, {expressions: ['lalallalal']}, 'measure');
  const mRes5 = calStatus(staticMeasure, {expressions: ['lalallalal', 'avg', 'min']}, 'measure');

  expect(dRes0).toBe('add');
  expect(dRes1).toBe('selected');
  expect(dRes2).toBe('selectBin');
  expect(dRes3).toBe('selected');
  expect(dRes4).toBe('selectBin');

  expect(mRes0).toBe('selected');
  expect(mRes1).toBe('selected');
  expect(mRes2).toBe('selected');
  expect(mRes3).toBe('selectExpression');
  expect(mRes4).toBe('selected');
  expect(mRes5).toBe('selectExpression');
});
test('measureUsePopUp', () => {
  const textMeasure = {value: 'ahhhh', expression: 'unique', type: 'text'};
  const numMeasure = {expression: 'avg', value: 'ahhhh'};
  const recordMeasure = {expression: 'count', value: '*', isRecords: true};

  const textRes = measureUsePopUp(textMeasure);
  const numRes = measureUsePopUp(numMeasure);
  const recordRes = measureUsePopUp(recordMeasure);

  expect(textRes).toBe(false);
  expect(numRes).toBe(true);
  expect(recordRes).toBe(false);
});
test('filterColumns', () => {
  const res = filterColumns('', columns);
  const res1 = filterColumns('asdasdasdasdasdas', columns);
  const res2 = filterColumns('build', columns);
  const res3 = filterColumns('fare', columns);
  const res4 = filterColumns('amount', columns);

  expect(res.length).toBe(16);
  expect(res1.length).toBe(0);
  expect(res2.length).toBe(4);
  expect(res3.length).toBe(1);
  expect(res4.length).toBe(3);
});
test('getDefaultTitle', () => {
  const res = getDefaultTitle({label: 'a', isRecords: false, expression: 'b'});
  const res1 = getDefaultTitle({label: 'b', isRecords: true, expression: 'c'});

  expect(res).toStrictEqual({
    expression: 'b',
    label: 'a',
  });
  expect(res1).toStrictEqual({
    expression: 'count',
    label: 'b',
  });
});
