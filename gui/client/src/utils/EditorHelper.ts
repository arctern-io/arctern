import {WIDGET, COLUMN_TYPE} from './Consts';
import {
  Measure,
  Dimension,
  CurrSetting,
  DimensionSetting,
  MeasureSetting,
  WidgetConfig,
  Setting,
  Params,
  QueryType,
} from '../types';
import {Status} from '../types/Editor';
import {getColType} from './ColTypes';
import {isValidValue} from './Helpers';
import {DEFAULT_ZOOM, DefaultMapTheme} from '../widgets/Utils/Map';
import {DEFAULT_SORT} from '../components/settingComponents/Sort';

export const QueryCount = 10;

type ReadyState = {
  isReady: boolean;
  lacks: any[];
};
const _isItemsReady = (
  items: Array<Dimension | Measure>,
  settings: Array<DimensionSetting | MeasureSetting>,
  readyStatus: ReadyState = {isReady: false, lacks: []}
): ReadyState => {
  const firstSetting = settings[0] || {type: 'noneRequired'};
  switch (firstSetting.type) {
    case 'requiedOneAtLeast':
      items.length > 0
        ? (readyStatus.isReady = true)
        : readyStatus.lacks.push({
            type: 'requiedOneAtLeast',
            key: '',
            short: firstSetting.short || '',
          });
      break;
    case 'required':
      readyStatus.isReady = true;
      settings.forEach((setting: DimensionSetting | MeasureSetting) => {
        if (setting.type === 'required') {
          const isExist = items.some((item: Dimension | Measure) => item.as === setting.key);
          if (!isExist) {
            readyStatus.isReady = false;
            readyStatus.lacks.push(setting);
          }
        }
      });
      break;
    case 'requiedOneDimensionOrMeasureAtLeast':
      readyStatus.isReady = true;
      if (items.length === 0) {
        readyStatus.isReady = false;
        readyStatus.lacks.push({name: '1'});
      }
      break;
    case 'noneRequired':
    case 'any':
    default:
      readyStatus.isReady = true;
      break;
  }
  return readyStatus;
};
export const isReadyToRender = (config: WidgetConfig, settings: Setting) => {
  const {dimensions, measures} = config;
  const dimensionsSetting = settings.dimensions;
  const measuresSetting = settings.measures;

  const sourceReady = {isReady: isValidValue(config.source)};
  const dimensionsReady: ReadyState = _isItemsReady(dimensions, dimensionsSetting);
  const measuresReady: ReadyState = _isItemsReady(measures, measuresSetting);
  // special handle when dimensions and measures ready rely on each other
  const firstDimensionSetting = dimensionsSetting[0] || {type: 'noneRequired'};
  if (
    firstDimensionSetting.type === 'requiedOneDimensionOrMeasureAtLeast' &&
    dimensions.length + measures.length > 0
  ) {
    dimensionsReady.isReady = true;
    measuresReady.isReady = true;
  }
  return {
    sourceReady,
    dimensionsReady,
    measuresReady,
  };
};

export const isRecordExist = (config: WidgetConfig): boolean => {
  return config.measures.some((m: Measure) => m.isRecords);
};

export type Column = {
  col_name: string;
  data_type: string;
};
export const getValidColumns = (columns: Column[] = [], columnTypes: COLUMN_TYPE[]): Column[] => {
  return columns.filter((column: Column) => {
    return columnTypes.some((validType: COLUMN_TYPE) => validType === getColType(column.data_type));
  });
};

export const genRangeQuery = (colName: string, source: string): Params => {
  let v = colName;
  const sql = `SELECT MIN(${v}) AS minimum, MAX(${v}) AS maximum FROM ${source}`;
  return {sql, type: QueryType.sql};
};

// TODO: delete later;
const _initConfig = (widgetType: string, config: any) => {
  const initConfig: any = {
    id: config.id || '',
    type: widgetType,
    title: config.title || '',
    source: config.source,
    layout: config.layout,
    dimensions: [],
    measures: [],
    colorItems: [],
    filter: {},
    selfFilter: {},
  };
  return initConfig;
};
// TODO: delete later;
export const convertConfig = (config: any, widgetType: string) => {
  // get common config like dimensions, measures, layout...
  const initConfig = _initConfig(widgetType, config);

  // add special attribute of each widget
  switch (widgetType) {
    case WIDGET.LINECHART:
      initConfig.isShowRange = false;
      break;
    case WIDGET.BARCHART:
      break;
    case WIDGET.PIECHART:
      break;
    case WIDGET.STACKEDBARCHART:
      initConfig.stackType = 'vertical';
      initConfig.sort = {
        name: '',
        order: 'descending',
      };
      break;
    case WIDGET.TABLECHART:
      initConfig.limit = 50;
      initConfig.sort = DEFAULT_SORT;
      break;
    case WIDGET.SCATTERCHART:
      initConfig.width = 850;
      initConfig.height = 400;
      initConfig.pointSize = 3;
      initConfig.points = 1000000;
      initConfig.popupItems = [];
      initConfig.isServerRender = true;
      break;
    case WIDGET.POINTMAP:
      initConfig.width = 850;
      initConfig.height = 400;
      initConfig.isServerRender = true;
      initConfig.mapTheme = DefaultMapTheme.value;
      initConfig.zoom = DEFAULT_ZOOM;
      initConfig.pointSize = 3;
      initConfig.points = 1000000;
      initConfig.popupItems = [];
      break;
    case WIDGET.GEOHEATMAP:
      initConfig.width = 810;
      initConfig.height = 465;
      initConfig.isServerRender = true;
      initConfig.colorKey = 'green_yellow_red';
      initConfig.mapTheme = DefaultMapTheme.value;
      initConfig.zoom = DEFAULT_ZOOM;
      break;
    case WIDGET.CHOROPLETHMAP:
      initConfig.width = 850;
      initConfig.height = 400;
      initConfig.isServerRender = true;
      initConfig.mapTheme = DefaultMapTheme.value;
      initConfig.zoom = DEFAULT_ZOOM;
      break;
    default:
      break;
  }
  return initConfig;
};

export const calStatus = (
  item: Dimension | Measure,
  currentSetting: CurrSetting,
  selectorType: 'dimension' | 'measure'
): Status => {
  const {value, type} = item;
  switch (selectorType) {
    case 'dimension':
      const {isNotUseBin} = currentSetting as DimensionSetting;
      const selectBin = !isNotUseBin && type !== COLUMN_TYPE.TEXT;
      return value ? (selectBin ? Status.SELECT_BIN : Status.SELECTED) : Status.ADD;
    case 'measure':
      const {expressions} = currentSetting as MeasureSetting;
      const selectExpression = measureUsePopUp(item as Measure) && settingUsePopUp(expressions);
      return value ? (selectExpression ? Status.SELECT_EXPRESSION : Status.SELECTED) : Status.ADD;
    default:
      return Status.SELECTED;
  }
};

export const settingUsePopUp = (expressions: string[]): boolean => {
  return expressions.length > 1;
};

export const measureUsePopUp = (measure: Measure | undefined): boolean => {
  if (!measure) {
    return false;
  }
  const {value, type, isRecords = false} = measure;
  return !!value && type !== 'text' && !isRecords;
};

let timeout: any = '';
export const delayRunFunc = (params: any, func: Function, time: number) => {
  if (timeout) {
    clearTimeout(timeout);
  }
  timeout = setTimeout(() => {
    func(params);
  }, time);
  const r = () => {
    clearTimeout(timeout);
  };
  return r;
};

export const genEffectClickOutside = (
  container: HTMLElement | null,
  callback: Function = () => {},
  params: any
) => {
  // root is the root Dom of whole document, in case there were float elements outside root like selectOpts in Bin.tsx, add !root.contains(target)
  // callback should be attribute of document not special object, in case get Error: Illigal invocaton
  return (e: any) => {
    const _container = container || document.createElement('div');
    const root = document.getElementById('root') || document.createElement('div');
    const target = e.target;
    if (!root.contains(target)) {
      return;
    }
    if (!_container.contains(target)) {
      callback(params);
    }
  };
};

export const filterColumns = (text: string | undefined, opts: Column[]) => {
  if (!text) {
    return opts;
  }
  const regex = new RegExp(text, 'i');
  return opts.filter((item: Column) => regex.test(item.col_name));
};

// Helper for Slider component
const _validValueGetter = (val: number, range: number[]) => {
  const [min, max] = range;
  if (val < min) {
    return min;
  }
  if (val > max) {
    return max;
  }
  return val;
};
const _delayPeriod = 300;
export const changeInputBox = ({
  e,
  range,
  immediateCallback = () => {},
  delayCallback = () => {},
  delayPeriod = _delayPeriod,
}: any) => {
  const val = e.target.value * 1;
  immediateCallback(val);
  const _val = _validValueGetter(val, range);
  delayRunFunc(_val, delayCallback, delayPeriod);
};

export const changeSlider = ({
  val,
  immediateCallback,
  delayCallback,
  delayPeriod = _delayPeriod,
}: any) => {
  val = val * 1;
  immediateCallback(val);
  delayRunFunc(val, delayCallback, delayPeriod);
};

const _validMinGetter = (val: number, range: number[], step: number) => {
  const [min, max] = range;
  if (val < min) {
    return [min, max];
  }
  if (val >= max) {
    return [max - step, max];
  }
  return [val, max];
};

const _validMaxGetter = (val: number, range: number[], step: number) => {
  const [min, max] = range;
  if (val <= min) {
    return [min, min + step];
  }
  if (val > max) {
    return [min, max];
  }
  return [min, val];
};

export const changeRangeSliderInputBox = ({
  val,
  range,
  step = 1,
  target,
  immediateCallback,
  delayCallback,
  delayPeriod = _delayPeriod,
}: any) => {
  immediateCallback(val);
  const validRange =
    target === 'min' ? _validMinGetter(val, range, step) : _validMaxGetter(val, range, step);
  delayRunFunc({validRange, target}, delayCallback, delayPeriod);
};

export const changeRangeSlider = ({
  val,
  immediateCallback,
  delayCallback,
  delayPeriod = _delayPeriod,
}: any) => {
  immediateCallback(val);
  delayRunFunc(val, delayCallback, delayPeriod);
};

export const getDefaultTitle = (measure: Measure) => {
  const {label, isRecords, expression} = measure;
  return {
    expression: isRecords ? 'count' : expression,
    label,
  };
};
