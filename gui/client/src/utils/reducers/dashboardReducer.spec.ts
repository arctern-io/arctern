import {dashboardReducer} from './dashboardReducer';
import {DASH_ACTIONS} from '../Consts';

test('Dashboard Reducers', () => {
  const notExistType = dashboardReducer(null as any, {type: 'not exist'} as any);
  expect(notExistType).toBeNull();

  let updateDashboard = dashboardReducer(
    null as any,
    {
      type: DASH_ACTIONS.UPDATE,
      payload: {id: 1, title: 'dashboard'},
    } as any
  );
  expect(updateDashboard).toEqual({id: 1, title: 'dashboard'});

  updateDashboard = dashboardReducer({id: 1, title: 'dashboard'} as any, {
    type: DASH_ACTIONS.UPDATE,
    payload: {id: 1, title: 'dashboard2', demo: true} as any,
  });
  expect(updateDashboard).toEqual({id: 1, title: 'dashboard2', demo: true});

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
  ];
  const updateConfigs = dashboardReducer(
    {id: 1, title: 'dashboard'} as any,
    {
      type: DASH_ACTIONS.UPDATE_CONFIGS,
      payload: configs,
    } as any
  );
  expect(updateConfigs).toEqual({id: 1, title: 'dashboard', configs});

  const updateTitle = dashboardReducer(
    {id: 1, title: 'dashboard'} as any,
    {
      type: DASH_ACTIONS.UPDATE_TITLE,
      payload: 'New Dashboad',
    } as any
  );
  expect(updateTitle).toEqual({id: 1, title: 'New Dashboad'});
});
