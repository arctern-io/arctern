import {CONFIG} from '../../utils/Consts';
import {WidgetConfig, Measure, Dimension} from '../../types';
import {cloneObj} from '../Helpers';

export type ConfigAction =
  | {type: CONFIG.UPDATE; payload: WidgetConfig}
  | {type: CONFIG.DEL_ATTR; payload: string[]}
  | {type: CONFIG.REPLACE_ALL; payload: WidgetConfig}
  | {type: CONFIG.UPDATE_TITLE; payload: string}
  | {type: CONFIG.ADD_DIMENSION; payload: {id: string; dimension: Dimension}}
  | {type: CONFIG.DEL_DIMENSION; payload: {id: string; dimension: Dimension}}
  | {type: CONFIG.ADD_MEASURE; payload: Measure}
  | {type: CONFIG.DEL_MEASURE; payload: Measure}
  | {type: CONFIG.ADD_SELF_FILTER; payload: any}
  | {type: CONFIG.DEL_SELF_FILTER; payload: {id: string; filterKeys: string[]}}
  | {type: CONFIG.ADD_FILTER; payload: WidgetConfig}
  | {type: CONFIG.DEL_FILTER; payload: {id: string; filterKeys: string[]}}
  | {type: CONFIG.CLEAR_FILTER; payload: WidgetConfig}
  | {type: CONFIG.ADD_SORT; payload: WidgetConfig}
  | {type: CONFIG.ADD_COLORITEMS; payload: any}
  | {type: CONFIG.DEL_COLORITEMS; payload: any}
  | {type: CONFIG.CLERAR_COLORITEMS; payload: WidgetConfig}
  | {type: CONFIG.ADD_COLORKEY; payload: WidgetConfig}
  | {type: CONFIG.DEL_COLORKEY; payload: WidgetConfig}
  | {type: CONFIG.ADD_RULER; payload: WidgetConfig}
  | {type: CONFIG.DEL_RULER; payload: WidgetConfig}
  | {type: CONFIG.ADD_RULERBASE; payload: WidgetConfig}
  | {type: CONFIG.DEL_RULERBASE; payload: WidgetConfig}
  | {type: CONFIG.ADD_POPUP_ITEM; payload: string}
  | {type: CONFIG.DEL_POPUP_ITEM; payload: string}
  | {type: CONFIG.ADD_LIMIT; payload: {id: string; limit: number}}
  | {type: CONFIG.UPDATE_POINTS; payload: number}
  | {type: CONFIG.UPDATE_POINT_SIZE; payload: number}
  | {type: CONFIG.ADD_MAPTHEME; payload: WidgetConfig}
  | {type: CONFIG.ADD_STACKTYPE; payload: string}
  | {type: CONFIG.CHANGE_IS_AREA; payload: boolean}
  | {type: CONFIG.UPDATE_AXIS_RANGE; payload: WidgetConfig};

export const singleConfigHandler = (config: WidgetConfig, action: ConfigAction) => {
  let copiedConfig = cloneObj(config);
  // console.info(CONFIG[action.type]);
  action.payload && delete action.payload.id;
  switch (action.type) {
    case CONFIG.UPDATE:
      return {...copiedConfig, ...action.payload};
    case CONFIG.DEL_ATTR:
      action.payload.forEach((a: string) => delete copiedConfig[a]);
      return copiedConfig;
    case CONFIG.REPLACE_ALL:
      return {...action.payload, id: config.id};
    case CONFIG.UPDATE_TITLE:
      copiedConfig.title = action.payload;
      return copiedConfig;
    case CONFIG.ADD_DIMENSION:
      copiedConfig.dimensions = addByAs(action.payload.dimension, copiedConfig.dimensions);
      return copiedConfig;
    case CONFIG.DEL_DIMENSION:
      copiedConfig.dimensions = deleteByAs(action.payload.dimension, copiedConfig.dimensions);
      return copiedConfig;
    case CONFIG.ADD_MEASURE:
      copiedConfig.measures = addByAs(action.payload, copiedConfig.measures);
      return copiedConfig;
    case CONFIG.DEL_MEASURE:
      copiedConfig.measures = deleteByAs(action.payload, copiedConfig.measures);
      return copiedConfig;
    // todo should be same with ADD_FILTER
    case CONFIG.ADD_SELF_FILTER:
      copiedConfig.selfFilter = {...copiedConfig.selfFilter, ...action.payload};
      return copiedConfig;
    case CONFIG.DEL_SELF_FILTER:
      copiedConfig.selfFilter = deleteFilter(action.payload.filterKeys, copiedConfig.selfFilter);
      return copiedConfig;
    case CONFIG.ADD_FILTER:
      copiedConfig.filter = {...copiedConfig.filter, ...action.payload};
      return copiedConfig;
    case CONFIG.DEL_FILTER:
      copiedConfig.filter = deleteFilter(action.payload.filterKeys, copiedConfig.filter);
      return copiedConfig;
    case CONFIG.CLEAR_FILTER:
      copiedConfig.filter = {};
      return copiedConfig;
    case CONFIG.ADD_SORT:
      copiedConfig.sort = action.payload;
      return copiedConfig;
    case CONFIG.ADD_COLORITEMS:
      copiedConfig.colorItems = addColorItems(action.payload, copiedConfig.colorItems);
      return copiedConfig;
    case CONFIG.DEL_COLORITEMS:
      copiedConfig.colorItems = deleteColorItems(action.payload, copiedConfig.colorItems);
      return copiedConfig;
    case CONFIG.CLERAR_COLORITEMS:
      copiedConfig.colorItems = [];
      return copiedConfig;
    case CONFIG.ADD_COLORKEY:
      copiedConfig.colorKey = action.payload;
      return copiedConfig;
    case CONFIG.DEL_COLORKEY:
      delete copiedConfig.colorKey;
      return copiedConfig;
    case CONFIG.ADD_RULER:
      copiedConfig.ruler = action.payload;
      return copiedConfig;
    case CONFIG.DEL_RULER:
      delete copiedConfig.ruler;
      return copiedConfig;
    case CONFIG.ADD_RULERBASE:
      copiedConfig.rulerBase = action.payload;
      return copiedConfig;
    case CONFIG.DEL_RULERBASE:
      delete copiedConfig.rulerBase;
      return copiedConfig;
    case CONFIG.ADD_POPUP_ITEM:
      copiedConfig.popupItems = copiedConfig.popupItems || [];
      action.payload && copiedConfig.popupItems.push(action.payload);
      return copiedConfig;
    case CONFIG.DEL_POPUP_ITEM:
      copiedConfig.popupItems = copiedConfig.popupItems.filter((p: string) => p !== action.payload);
      return copiedConfig;
    case CONFIG.ADD_LIMIT:
      copiedConfig.limit = action.payload.limit;
      return copiedConfig;
    case CONFIG.UPDATE_POINTS:
      copiedConfig.points = action.payload;
      return copiedConfig;
    case CONFIG.UPDATE_POINT_SIZE:
      copiedConfig.pointSize = action.payload;
      return copiedConfig;
    case CONFIG.ADD_MAPTHEME:
      copiedConfig.mapTheme = action.payload;
      return copiedConfig;
    case CONFIG.ADD_STACKTYPE:
      copiedConfig.stacktype = action.payload;
      return copiedConfig;
    case CONFIG.CHANGE_IS_AREA:
      copiedConfig.isArea = action.payload;
      return copiedConfig;
    case CONFIG.UPDATE_AXIS_RANGE:
      const xMeasure = copiedConfig.measures.find((m: Measure) => m.as === 'x');
      const yMeasure = copiedConfig.measures.find((m: Measure) => m.as === 'y');
      if (!xMeasure || !yMeasure) {
        throw new Error('xMeasure and yMeasure must exist.');
      }
      xMeasure.domain = action.payload.x;
      yMeasure.domain = action.payload.y;

      return copiedConfig;
    default:
      return config;
  }
};

const localConfigReducer = (config: WidgetConfig, action: ConfigAction) => {
  return singleConfigHandler(config, action);
};

export default localConfigReducer;

type DorM = Dimension | Measure;
type DorMHandler = (item: DorM, targets: DorM[]) => DorM[];

export const addByAs: DorMHandler = (item, targets) => {
  const index = targets.findIndex(t => t.as === item.as);
  index < 0 ? targets.push(item) : targets.splice(index, 1, item);
  return targets;
};
export const deleteByAs: DorMHandler = (item, targets) => {
  return targets.filter((t: any) => t.as !== item.as);
};

export const deleteFilter = (keys: string[], filter: any = {}) => {
  const copyFilter = cloneObj(filter);
  keys.forEach((k: string) => delete copyFilter[k]);
  return copyFilter;
};
export const addColorItems = (newColorItems: any[] = [], originColorItems: any[] = []) => {
  newColorItems.forEach((c: any) => {
    addByAs(c, originColorItems);
  });
  return originColorItems;
};
export const deleteColorItems = (colorItems: any[] = [], originColorItems: any = []) => {
  const colorItemsAs = colorItems.map((c: any) => c.as);
  return originColorItems.filter((c: any) => !colorItemsAs.includes(c.as));
};
