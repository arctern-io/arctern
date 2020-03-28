import {TimeBin} from './Time';

export enum WIDGET {
  NUMBERCHART = 'NumberChart',
  FILTERWIDGET = 'FilterWidget',
  PIECHART = 'PieChart',
  BARCHART = 'BarChart',
  STACKEDBARCHART = 'StackedBarChart',
  TABLECHART = 'TableChart',
  HEATCHART = 'HeatChart',
  BUBBLECHART = 'BubbleChart',
  POINTMAP = 'PointMap',
  GEOHEATMAP = 'GeoHeatMap',
  SCATTERCHART = 'ScatterChart',
  CHOROPLETHMAP = 'ChoroplethMap',
  TEXTCHART = 'TextChart',
  HISTOGRAMCHART = 'HistogramChart',
  LINECHART = 'LineChart',
  COMBOCHART = 'ComboChart',
  LINEMAP = 'LineMap',
  MAPCHART = 'MapChart',
  AREACHART = 'AreaChart',
  RANGECHART = 'RangeChart',
  MAPBOX = 'MapBox',
}
export enum RequiredType {
  REQUIRED = 'required',
  REQUIRED_ONE_AT_LEAST = 'requiedOneAtLeast',
  REQUIRED_ONE_DIMENSION_OR_MEASURE_AT_LEAST = 'requiedOneDimensionOrMeasureAtLeast',
  ANY = 'any',
  OPTION = 'option',
  NONE_REQUIRED = 'noneRequired',
}
export enum COLUMN_TYPE {
  TEXT = 'text',
  NUMBER = 'number',
  DATE = 'date',
  UNKNOWN = 'unknown',
}

export const DEFAULT_CHART = WIDGET.NUMBERCHART;
export enum SORT_ORDER {
  ASC = 'ascending',
  DESC = 'descending',
}
export enum DASH_ACTIONS {
  UPDATE,
  DEL,
  UPDATE_CONFIGS,
  UPDATE_TITLE,
  LAYOUT_CHANGE = 'LayoutChange',
  CURR_USED_LAYOUT_CHANGE = 'CurrUsedLayoutChange',
}

export enum CONFIG {
  UPDATE,
  LOCAL_UPDATE,
  DEL,
  DEL_ATTR,
  REPLACE_ALL,
  UPDATE_TITLE,
  ADD_DIMENSION,
  DEL_DIMENSION,
  ADD_MEASURE,
  DEL_MEASURE,

  ADD_SELF_FILTER,
  DEL_SELF_FILTER,
  ADD_FILTER,
  DEL_FILTER,
  CLEAR_FILTER,
  CLEARALL_FILTER,

  ADD_SORT,
  DEL_SORT,
  ADD_COLORITEMS,
  DEL_COLORITEMS,
  CLERAR_COLORITEMS,
  ADD_COLORKEY,
  DEL_COLORKEY,
  ADD_RULER,
  DEL_RULER,
  ADD_RULERBASE,
  DEL_RULERBASE,
  ADD_POPUP_ITEM,
  DEL_POPUP_ITEM,
  ADD_STACKTYPE,
  ADD_LIMIT,
  UPDATE_POINTS,
  UPDATE_POINT_SIZE,
  ADD_MAPTHEME,
  CHANGE_IS_AREA,
  UPDATE_AXIS_RANGE,
}

export enum MODE {
  ADD = 'Add',
  NORMAL = 'normal',
  EDIT = 'edit',
}
enum DefaultExpression {
  avg = 'avg',
  min = 'min',
  max = 'max',
  sum = 'sum',
  unique = 'unique',
  stddev = 'stddev',
}
export type DefaultExpressionOption = {
  label: string;
  value: DefaultExpression;
};
export type CustomExpressOption = {
  label: string;
  value: string;
};
export const EXPRESSION_OPTIONS: DefaultExpressionOption[] = [
  {label: 'AVG', value: DefaultExpression.avg},
  {label: 'MIN', value: DefaultExpression.min},
  {label: 'MAX', value: DefaultExpression.max},
  {label: 'SUM', value: DefaultExpression.sum},
  {label: '# UNIQUE', value: DefaultExpression.unique},
  {label: 'STDDEV', value: DefaultExpression.stddev},
];

export const OUT_OUT_CHART = -999999999;

export const defaultChartMousePos: any = {
  x: OUT_OUT_CHART,
  y: -1,
  xV: null,
  yV: null,
};

export const MAX_YTICK_NUM = 10;
export const MIN_TICK_HEIGHT = 16;
export const NO_DATA = `NO DATA`;
export enum DIALOG_MODE {
  INFO = 'INFO',
  CONFIRM = 'CONFIRM',
}

export const DefaultDimension = {
  name: '',
  format: 'auto',
  type: COLUMN_TYPE.NUMBER,
  label: '',
  value: '',
  as: '',
  isBinned: false,
  extract: false,
  isCustom: false,
  min: 0,
  max: 0,
  currMin: 0,
  currMax: 0,
  extent: [],
  staticRange: [],
  timeBin: TimeBin.CENTURY,
};
