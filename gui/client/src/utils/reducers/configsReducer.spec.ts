import configsReducer, {_updateSingleConfig} from './configsReducer';
import {CONFIG, DASH_ACTIONS} from '../Consts';
import {cloneObj} from '../Helpers';
const configs = [
  {
    id: '1',
    title: '',
    filter: {range: []},
    selfFilter: {range: []},
    type: 'HistogramChart',
    source: 'nyc_taxi',
    dimensions: [],
    measures: [],
    layout: {
      i: '1',
      h: 100,
      w: 100,
    },
  },
  {
    id: '2',
    title: '',
    filter: {},
    selfFilter: {},
    type: 'BarChart',
    source: 'nyc_taxi',
    dimensions: [],
    measures: [],
    layout: {
      i: '2',
      h: 200,
      w: 200,
    },
  },
];

test('Configs Reducer _updateSingleConfig', () => {
  const copyConfigs = cloneObj(configs);
  const result = _updateSingleConfig(copyConfigs, {id: '2', title: 'replace'} as any);
  expect(result.find(v => v.id === '2')).toEqual({id: '2', title: 'replace'});
  const addResult = _updateSingleConfig(copyConfigs, {id: '3', title: 'new'} as any);
  expect(addResult.length).toEqual(3);
  expect(addResult.map(v => v.id)).toEqual(['1', '2', '3']);
});

test('Configs Reducer', () => {
  const notExistType = configsReducer(configs as any, {type: 'not exist'} as any);
  expect(notExistType).toEqual(configs);

  const updateSameIdConfig = configsReducer(
    configs as any,
    {
      type: CONFIG.UPDATE,
      payload: {id: '1', type: 'LineChart', title: 'line'},
    } as any
  );
  expect(updateSameIdConfig.length).toEqual(2);
  expect(updateSameIdConfig).toContainEqual({id: '1', type: 'LineChart', title: 'line'});

  const addNewConfig = configsReducer(
    configs as any,
    {
      type: CONFIG.UPDATE,
      payload: {id: '3', type: 'LineChart', title: 'line'},
    } as any
  );
  expect(addNewConfig.length).toEqual(3);
  expect(addNewConfig).toContainEqual({id: '3', type: 'LineChart', title: 'line'});

  const deleteConfigById = configsReducer(
    configs as any,
    {
      type: CONFIG.DEL,
      payload: {id: '1'},
    } as any
  );
  expect(deleteConfigById.length).toEqual(1);

  const clearAllfilters = configsReducer(
    configs as any,
    {
      type: CONFIG.CLEARALL_FILTER,
      payload: configs,
    } as any
  );
  clearAllfilters.forEach((config: any) => {
    expect(config.filter).toEqual({});
  });

  const newLayout = [
    {
      i: '1',
      w: 1,
      h: 1,
    },
    {
      i: '2',
      w: 2,
      h: 2,
    },
  ];
  const layoutChange = configsReducer(
    configs as any,
    {
      type: DASH_ACTIONS.LAYOUT_CHANGE,
      payload: newLayout,
    } as any
  );
  layoutChange.forEach((config: any) => {
    const {id} = config;
    const layout = newLayout.find(v => v.i === id);
    expect(config.layout).toEqual(layout);
  });

  const newConfigs = [
    {
      id: '1',
      type: 'HistogramChart',
      layout: {
        i: '1',
        h: 100,
        w: 100,
      },
    },
  ];
  const curLayoutChange = configsReducer(
    configs as any,
    {
      type: DASH_ACTIONS.CURR_USED_LAYOUT_CHANGE,
      payload: newConfigs,
    } as any
  );

  expect(curLayoutChange).toEqual(newConfigs);

  const addFilterConfigs = configsReducer(
    configs as any,
    {
      type: CONFIG.ADD_FILTER,
      payload: {
        building: {
          expr: {
            field: 'buildingid_dropoff',
            left: 106097.75,
            right: 212195.5,
            type: 'between',
          },
          type: 'filter',
        },
        id: '1',
      },
    } as any
  );

  expect(addFilterConfigs[0]).toEqual({
    id: '1',
    title: '',
    filter: {
      building: {
        expr: {
          field: 'buildingid_dropoff',
          left: 106097.75,
          right: 212195.5,
          type: 'between',
        },
        type: 'filter',
      },
    },
    selfFilter: {range: []},
    type: 'HistogramChart',
    source: 'nyc_taxi',
    dimensions: [],
    measures: [],
    layout: {
      i: '1',
      h: 100,
      w: 100,
    },
  });
});
