import localConfigReducer, {
  addByAs,
  deleteByAs,
  deleteColorItems,
  deleteFilter,
  addColorItems,
} from './localConfigReducer';
import {CONFIG} from '../../utils/Consts';

const config = {
  id: '1',
  title: '',
  filter: {range: []},
  selfFilter: {range: []},
  type: 'HistogramChart',
  source: 'nyc_taxi',
  dimensions: [
    {
      as: 'x',
      isBinned: false,
    },
  ],
  measures: [],
  layout: {
    i: '1',
    h: 100,
    w: 100,
  },
};

test('localConfigReducer addByAs', () => {
  const result = addByAs({as: 'x', format: 'auto'} as any, [{as: 'x'}] as any);
  expect(result.length).toEqual(1);
  expect(result[0]).toEqual({as: 'x', format: 'auto'});

  const addResult = addByAs({as: 'y'} as any, [{as: 'x'}] as any);
  expect(addResult.length).toEqual(2);
});

test('localConfigReducer deleteByAs', () => {
  const result = deleteByAs({as: 'x'} as any, [{as: 'x'}, {as: 'y'}] as any);
  expect(result.length).toEqual(1);
  expect(result[0]).toEqual({as: 'y'});
});

test('localConfigReducer deleteFilter', () => {
  const filters = {range: [], color: [], text: []};
  let result = deleteFilter(['range'], filters);
  expect(result).toEqual({color: [], text: []});
  result = deleteFilter(['color', 'text'], filters);
  expect(result).toEqual({range: []});
});

test('localConfigReducer addColorItems', () => {
  const colorItems = [{as: 'x', color: '#fff'}];
  const originColor = [
    {as: 'y', color: 'red'},
    {as: 'x', color: 'blue'},
  ];
  const replace = addColorItems(colorItems, originColor);
  expect(replace).toEqual([
    {as: 'y', color: 'red'},
    {as: 'x', color: '#fff'},
  ]);

  const addResult = addColorItems([{as: 'value', color: '#000'}], originColor);
  expect(addResult.length).toEqual(3);
});

test('localConfigReducer deleteColorItems', () => {
  const colorItems = [{as: 'x'}, {as: 'z'}];
  const originColor = [
    {as: 'y', color: 'red'},
    {as: 'x', color: 'blue'},
  ];
  const deleteResult = deleteColorItems(colorItems, originColor);
  expect(deleteResult).toEqual([{as: 'y', color: 'red'}]);
});

test('Local Config Reducer', () => {
  const notExistType = localConfigReducer(config as any, {type: 'not exist'} as any);
  expect(notExistType).toEqual(config);

  const updateConfig = localConfigReducer(
    config as any,
    {
      type: CONFIG.UPDATE,
      payload: {id: '1', title: 'new', filter: {range: [1, 2]}, type: 'LineChart'},
    } as any
  );
  expect(updateConfig).toEqual({
    id: '1',
    title: 'new',
    filter: {range: [1, 2]},
    selfFilter: {range: []},
    type: 'LineChart',
    source: 'nyc_taxi',
    dimensions: [
      {
        as: 'x',
        isBinned: false,
      },
    ],
    measures: [],
    layout: {
      i: '1',
      h: 100,
      w: 100,
    },
  });

  const deleteConfigAttr = localConfigReducer(config as any, {
    type: CONFIG.DEL_ATTR,
    payload: ['filter', 'selfFilter'],
  });
  expect(deleteConfigAttr.filter).toBeUndefined();
  expect(deleteConfigAttr.selfFilter).toBeUndefined();

  const replaceAllConfig = localConfigReducer(
    config as any,
    {
      type: CONFIG.REPLACE_ALL,
      payload: {dimensions: [], measures: [], id: '2', source: 'nyc_taxi'},
    } as any
  );
  expect(replaceAllConfig).toEqual({dimensions: [], measures: [], id: '1', source: 'nyc_taxi'});

  const updateTitle = localConfigReducer(config as any, {
    type: CONFIG.UPDATE_TITLE,
    payload: 'new title',
  });
  expect(updateTitle.title).toEqual('new title');

  let updateDimensions = localConfigReducer(
    config as any,
    {
      type: CONFIG.ADD_DIMENSION,
      payload: {
        dimension: {
          as: 'x',
          isBinned: true,
        },
      },
    } as any
  );
  expect(updateDimensions.dimensions).toEqual([{as: 'x', isBinned: true}]);

  updateDimensions = localConfigReducer(
    config as any,
    {
      type: CONFIG.ADD_DIMENSION,
      payload: {
        dimension: {
          as: 'y',
          isBinned: false,
        },
      },
    } as any
  );
  expect(updateDimensions.dimensions).toEqual([
    {as: 'x', isBinned: false},
    {as: 'y', isBinned: false},
  ]);

  updateDimensions = localConfigReducer(updateDimensions, {
    type: CONFIG.DEL_DIMENSION,
    payload: {
      dimension: {
        as: 'x',
        isBinned: false,
      },
    },
  } as any);
  expect(updateDimensions.dimensions).toEqual([{as: 'y', isBinned: false}]);

  let updateMeasureConfig = localConfigReducer(
    config as any,
    {
      type: CONFIG.ADD_MEASURE,
      payload: {
        as: 'y',
        isBinned: false,
      },
    } as any
  );
  expect(updateMeasureConfig.measures).toEqual([{as: 'y', isBinned: false}]);

  updateMeasureConfig = localConfigReducer(updateMeasureConfig, {
    type: CONFIG.DEL_MEASURE,
    payload: {
      as: 'y',
    },
  } as any);
  expect(updateMeasureConfig.measures.length).toEqual(0);

  let updateFilter = localConfigReducer(
    config as any,
    {
      type: CONFIG.ADD_FILTER,
      payload: {color: '1'},
    } as any
  );
  expect(updateFilter.filter).toEqual({range: [], color: '1'});
  updateFilter = localConfigReducer(updateFilter, {
    type: CONFIG.DEL_FILTER,
    payload: {
      filterKeys: ['color', 'range'],
    },
  } as any);
  expect(updateFilter.filter).toEqual({});

  updateFilter = localConfigReducer(config as any, {type: CONFIG.CLEAR_FILTER} as any);
  expect(updateFilter.filter).toEqual({});

  let updateColorItems = localConfigReducer(config as any, {
    type: CONFIG.ADD_COLORITEMS,
    payload: [
      {as: 'VTS', colName: 'VendorID', color: '#37A2DA', label: 'VTS', value: 'VTS'},
      {as: 'CMT', color: '#fff'},
    ],
  });
  expect(updateColorItems.colorItems.length).toEqual(2);

  updateColorItems = localConfigReducer(updateColorItems, {
    type: CONFIG.DEL_COLORITEMS,
    payload: [{as: 'VTS'}],
  });
  expect(updateColorItems.colorItems.length).toEqual(1);

  let testConfig = {
    ...config,
    measures: [{as: 'x'}, {as: 'y'}],
  };
  let updateAxisRange = localConfigReducer(
    testConfig as any,
    {
      type: CONFIG.UPDATE_AXIS_RANGE,
      payload: {
        x: [0, 2],
        y: [1, 100],
      },
    } as any
  );

  expect(updateAxisRange.measures).toEqual([
    {as: 'x', domain: [0, 2]},
    {as: 'y', domain: [1, 100]},
  ]);
});
