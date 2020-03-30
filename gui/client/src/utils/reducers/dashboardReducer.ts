import {DASH_ACTIONS} from '../Consts';
import {Dashboard, WidgetConfig} from '../../types';
import {cloneObj} from '../Helpers';

export type DashboardAction =
  | {type: DASH_ACTIONS.UPDATE_CONFIGS; payload: WidgetConfig[]}
  | {type: DASH_ACTIONS.UPDATE_TITLE; payload: string}
  | {type: DASH_ACTIONS.UPDATE; payload: Dashboard};

export const dashboardReducer = (dashboard: Dashboard, action: DashboardAction) => {
  let copiedDashboard = cloneObj(dashboard);
  let copiedPayload = cloneObj(action.payload || {});
  switch (action.type) {
    case DASH_ACTIONS.UPDATE:
      return {...copiedDashboard, ...copiedPayload};

    case DASH_ACTIONS.UPDATE_CONFIGS:
      copiedDashboard.configs = copiedPayload;
      return copiedDashboard;

    case DASH_ACTIONS.UPDATE_TITLE:
      copiedDashboard.title = copiedPayload;
      return copiedDashboard;

    default:
      return dashboard;
  }
};
