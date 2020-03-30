import {exportCsv, exportJson} from './Export';
const config = {
  id: 'id_iqeo1mlxh8',
  title: 'test',
  filter: {},
  selfFilter: {},
  type: 'NumberChart',
  source: 'car_move',
  dimensions: [],
  measures: [
    {
      type: 'text',
      name: 'value',
      value: 'alarm',
      label: 'alarm',
      as: 'value',
      format: 'auto',
      isCustom: false,
      isRecords: false,
      expression: 'unique',
    },
  ],
  layout: {w: 10, h: 10, x: 0, y: 0, i: 'id_iqeo1mlxh8'},
  colorKey: '#37A2DA',
  isServerRender: false,
};
const data = [{value: 6}];

const dashboardJson = {
  id: 5,
  demo: false,
  title: 'DashboardJson-5',
  userId: 'infini',
  configs: [
    {
      id: 'id_iqeo1mlxh8',
      title: '',
      type: 'NumberChart',
      source: 'car_move',
    },
  ],
  createdAt: 'Tue Jan  7 14:39:18 2020',
  modifyAt: 'Tue Jan  7 14:39:18 2020',
  sources: ['car_move', 'nyc_taxi', 'wuxian_wifi'],
  sourceOptions: [1, 2, 3],
};
test('Export ', () => {
  const {encodedUri, filename} = exportCsv(config as any, data);
  expect(encodedUri).toEqual('data:text/csv;charset=utf-8,alarm%0D%0A6%0D%0A');
  expect(filename).toEqual('infini-export-car_move-test.csv');

  const {filename: dashboardFileName, csvContent} = exportJson(dashboardJson);
  expect(dashboardFileName).toEqual('infini-dashboard-DashboardJson-5.json');

  const copyJson = JSON.parse(JSON.stringify(dashboardJson));
  delete copyJson.sourceOptions;
  expect(csvContent).toEqual(
    `data:text/json;charset=utf-8,${encodeURIComponent(JSON.stringify(copyJson))}\r\n`
  );
});
