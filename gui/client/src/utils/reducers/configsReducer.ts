import {CONFIG, DASH_ACTIONS} from '../../utils/Consts';
import {WidgetConfig, Layout} from '../../types';
import {cloneObj} from '../Helpers';
import {
  singleConfigHandler,
  ConfigAction as SingleAction,
  ConfigAction,
} from './localConfigReducer';

export type ConfigsAction =
  | {type: CONFIG.UPDATE; payload: WidgetConfig}
  | {type: CONFIG.DEL; payload: WidgetConfig}
  | {type: CONFIG.CLEARALL_FILTER; payload: WidgetConfig[]}
  | {type: DASH_ACTIONS.LAYOUT_CHANGE; payload: Layout[]}
  | {type: DASH_ACTIONS.CURR_USED_LAYOUT_CHANGE; payload: WidgetConfig[]}
  | SingleAction;

const configsReducer = (configs: WidgetConfig[], action: ConfigsAction) => {
  let copiedConfigs = cloneObj(configs);
  switch (action.type) {
    case CONFIG.UPDATE:
      return _updateSingleConfig(copiedConfigs, action.payload);
    case CONFIG.DEL:
      return copiedConfigs.filter((config: WidgetConfig) => config.id !== action.payload.id);
    case CONFIG.CLEARALL_FILTER:
      return action.payload.map((config: WidgetConfig) => {
        config.filter = {};
        return config;
      });
    case DASH_ACTIONS.LAYOUT_CHANGE:
      const layouts = action.payload;
      layouts.forEach((layout: Layout) => {
        const id = layout.i;
        const index = configs.findIndex(config => config.id === id);
        copiedConfigs[index].layout = layout;
      });
      return copiedConfigs;
    // todo we should put logic here
    case DASH_ACTIONS.CURR_USED_LAYOUT_CHANGE:
      return action.payload;
    case CONFIG.ADD_FILTER:
    case CONFIG.DEL_FILTER:
    case CONFIG.CLEAR_FILTER:
    case CONFIG.UPDATE_AXIS_RANGE:
    case CONFIG.DEL_SELF_FILTER:
    case CONFIG.ADD_SORT:
    case CONFIG.ADD_LIMIT:
      const newConfig = configs.find((c: WidgetConfig) => c.id === action.payload.id) || configs[0];
      return _updateSingleConfig(
        copiedConfigs,
        singleConfigHandler(newConfig, action as ConfigAction)
      );
    default:
      return configs;
  }
};
export default configsReducer;

export const _updateSingleConfig = (configs: WidgetConfig[], config: WidgetConfig) => {
  const index = configs.findIndex((_config: WidgetConfig) => config.id === _config.id);
  // insert one at index or add one at the end
  index > -1 ? configs.splice(index, 1, config) : configs.push(config);
  return configs;
};
