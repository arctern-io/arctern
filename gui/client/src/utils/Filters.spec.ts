import {
  widgetFiltersGetter,
  getFilterLength,
  dimensionsDataToFilterExpr,
  orFilterGetter,
  andFilterGetter,
} from './Filters';

test('Filter widgetFiltersGetter', () => {
  let config = {
    type: 'HistogramChart',
    dimensions: [
      {
        name: 'x',
        format: 'auto',
        type: 'DATE',
        value: 'system_time',
        label: 'system_time',
        isBinned: true,
        extract: true,
        as: 'x',
        min: 1,
        max: 60,
        extent: ['Mon Oct 08 00:00:17 2018', 'Mon Oct 08 23:59:59 2018'],
        staticRange: ['Mon Oct 08 00:00:17 2018', 'Mon Oct 08 23:59:59 2018'],
        timeBin: 'minute',
        binningResolution: '1h',
        currMin: 1,
        currMax: 60,
      },
    ],
    filter: {},
    selfFilter: {},
    id: 'id_q9uj8h5ruy',
  };

  // no filter
  expect(widgetFiltersGetter(config as any).length).toEqual(0);

  config.filter = {
    range: {
      type: 'filter',
      expr: {
        type: 'between',
        originField: 'system_time',
        field: 'extract(minute from system_time)',
        left: 30,
        right: 43,
      },
    },
    color: {
      type: 'filter',
      expr: {
        originField: 'nope',
        field: 'nope',
      },
    },
  };
  const filters = widgetFiltersGetter(config as any);
  expect(filters.length).toEqual(1);
  expect(filters[0]?.name).toEqual('range');
});

test('Filters getFilterLength', () => {
  const filters = [{}, {range: {}}, {haha: {}}];
  expect(getFilterLength(filters as any)).toEqual(2);
});

test('Filters dimensionsDataToFilterExpr', () => {
  let dimensionsData: any = [
    {
      dimension: {
        format: 'auto',
        type: 'text',
        value: 'car_num',
        label: 'car_num',
        as: 'alarm_s1wio1x68ll',
      },
      data: ['B3X1G6'],
    },
  ];

  // andFilter: {id: 'andFilter', type: 'filter', expr: "car_num = 'B3X1G6'"}
  expect(dimensionsDataToFilterExpr(dimensionsData)).toEqual("car_num = 'B3X1G6'");

  dimensionsData = [
    ...dimensionsData,
    {
      dimension: {
        format: 'auto',
        type: 'text',
        value: 'price',
        label: 'price',
        as: 'alarm_s1wio1x68ll',
      },
      data: ['12'],
    },
  ];

  expect(dimensionsDataToFilterExpr(dimensionsData)).toEqual("car_num = 'B3X1G6' AND price = '12'");

  dimensionsData = [
    {
      dimension: {
        format: 'auto',
        type: 'DATE',
        value: 'system_time',
        label: 'system_time',
        isBinned: true,
        extract: false,
        as: 'system_time_j3pexub3bir',
        min: 'Mon Oct 08 00:00:17 2018',
        max: 'Mon Oct 08 23:59:59 2018',
        extent: ['Mon Oct 08 00:00:17 2018', 'Mon Oct 08 23:59:59 2018'],
        staticRange: ['Mon Oct 08 00:00:17 2018', 'Mon Oct 08 23:59:59 2018'],
        timeBin: 'hour',
        binningResolution: '1h',
      },
      data: ['Mon Oct 08 13:00:00 2018', 'Mon, 08 Oct 2018 05:59:59 GMT'],
    },
  ];

  expect(dimensionsDataToFilterExpr(dimensionsData)).toEqual(
    "system_time BETWEEN '2018-10-08T13:00:00' AND '2018-10-08T13:59:59'"
  );
});

test('Filters orFilterGetter', () => {
  const filters = {
    id_p3u9sqwn7zr: {type: 'filter', expr: "alarm = '1'"},
    id_7h811klsbfm: {type: 'filter', expr: "alarm = '2'"},
  };
  expect(orFilterGetter(filters as any)).toEqual({
    orFilter: {type: 'filter', expr: "(alarm = '1') OR (alarm = '2')"},
  });
});

test('Filters andFilterGetter', () => {
  const filters = {
    id_p3u9sqwn7zr: {type: 'filter', expr: "alarm = '1'"},
    id_7h811klsbfm: {type: 'filter', expr: "alarm = '2'"},
  };
  expect(andFilterGetter(filters as any)).toEqual({
    andFilter: {
      type: 'filter',
      expr: "alarm = '1' AND alarm = '2'",
    },
  });
});
