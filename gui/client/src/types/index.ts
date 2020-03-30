import React from 'react';
import {SORT_ORDER, MODE, WIDGET, RequiredType, COLUMN_TYPE} from '../utils/Consts';
import {DashboardAction} from '../utils/reducers/dashboardReducer';
import {Filter as coreFilter, Expression as coreExpression} from '../core/types';
import {TimeBin, ExtractBin} from '../utils/Time';

///////////// Database settings /////////////
export type DB_TYPE = {
  type: string;
  id: string;
  name: string;
};
///////////// Data and Query Types /////////////
// server side returned data
// it should be array of object
export type Data = any[];
// each key is config id, value is its data
export type DataCache = {[key: string]: Data};

export enum QueryType {
  sql = 'sql',
  point = 'point',
  heat = 'heat',
  choropleth = 'choropleth',
}
export type Params = {
  type: QueryType;
  sql: string;
  params?: any;
};
// a typical query type
export type Query = {
  // widget identity
  id: string;
  // sql and related params
  params: Params;
  // query time
  timestamp?: number;
};

// store request information
// it can be used as
type MetaInfo = {
  // widget identity
  id: string;
  // database sql
  sql: string;
  // is loading
  loading: boolean;
  // timestamp
  timestamp?: number;
};

// in a dashboard, every request's metainfo is stored in the Meta Map
export type Meta = {[key: string]: MetaInfo};

///////////// Dashboard Types /////////////
// Source is database table, string
export type Source = string;
// the Source Option is
// TODO: define more specific type for it
export type SourceOptions = any;
// Dashboard
export type Dashboard = {
  // dashboard id
  id: number;
  // is it a demo
  demo: boolean;
  // who is the creator
  userId: string;
  // dashboard title
  title: string;
  // dashboard sources
  sources: Source[];
  // dashboard sources options
  sourceOptions: SourceOptions;
  // which widgets are in this dashboard
  configs: WidgetConfig[];
  // other properties if have
  [propName: string]: any;
};

// dashboard page
export type DashboardProps = {
  dashboard: Dashboard;
  setDashboard: React.Dispatch<DashboardAction>;
};

///////////// WidgetConfig Settings /////////////
export type ConfigHandler<T = WidgetConfig> = (config: T) => T;
type BaseColSetting = {
  type: RequiredType;
  // column types
  columnTypes?: COLUMN_TYPE[];
  // dimension or measure key, the unique identity of dimension or measure in each config
  key?: string;
  // dimension or measure pre note
  // we will read `label_widgetEditor_requireLabel_${short}` as short or label itself if not existing
  short?: string;
  // fire on dimension or measure added
  onAdd?: Function;
  // fire on dimension or measure removed
  onDelete?: Function;
  // if we have multiple optional dimension or measure, we need some rule for labeling
  labelPlus?: string;
};
export type DimensionSetting = BaseColSetting & {
  expression?: string;
  // enable/disable bin
  isNotUseBin?: boolean;
};

export type InitMeasureSetting = BaseColSetting & {
  expressions?: string[];
  // is this measure as count(*)
  isRecords?: boolean;
};
export type MeasureSetting = BaseColSetting & {
  expressions: string[];
  // is this measure as count(*)
  isRecords?: boolean;
};

export type CurrSetting = DimensionSetting | MeasureSetting;

// widget setting
export type InitSetting<T = WidgetConfig> = {
  // widget Type
  type: string;
  // is this widget enabled
  enable?: boolean;
  // is widget rendered by server
  isServerRender?: boolean;
  // dimensions' setting
  dimensions: DimensionSetting[];
  // measures' setting
  measures: InitMeasureSetting[];
  // svg string
  icon: string;
  // configHandler, it will fired before generating crossfilter sql
  configHandler?: ConfigHandler<T>;
  // it will be fired after having sql generated
  onAfterSqlCreate?: Function;
};
export type Setting<T = WidgetConfig> = {
  // widget Type
  type: string;
  // is this widget enabled
  enable: boolean;
  // is widget rendered by server
  isServerRender: boolean;
  // dimensions' setting
  dimensions: DimensionSetting[];
  // measures' setting
  measures: MeasureSetting[];
  // svg string
  icon: string;
  // configHandler, it will fired before generating crossfilter sql
  configHandler: ConfigHandler<T>;
  // it will be fired after having sql generated
  onAfterSqlCreate: Function;
};
// all widgets Settings
export type WidgetSettings = {[key: string]: Setting};

///////////// WidgetConfig Layout /////////////
// react-grid-layout's configaration
export type Layout = {
  i: string;
  x: number;
  y: number;
  w: number;
  h: number;
  static: boolean;
  [propName: string]: any;
};

///////////// WidgetConfig Dimension /////////////
export type Dimension = {
  format: string;
  type: string;
  label: string;
  value: string;
  as: string;
  short?: string;
  isBinned?: boolean;
  extract?: boolean;
  min?: string | number;
  max?: string | number;
  extent?: Array<string | number>;
  staticRange?: Array<string | number>;
  timeBin?: TimeBin | ExtractBin;
  maxbins?: number;
  isCustom?: boolean;
  isNotUseBin?: boolean;
  options?: string[];
  expression?: string;
  binningResolution?: string;
};

///////////// WidgetConfig Measure /////////////
export type Measure = {
  format: string;
  type: string;
  label: string;
  value: string;
  as: string;
  short?: string;
  isCustom?: boolean;
  isRecords?: boolean;
  expression: string;
  domain?: [number, number];
  range?: [number, number] | number;
  staticDomain?: [number, number];
};
///////////// WidgetConfig Filters  /////////////
export type Filter = coreFilter;
type _Expression = string | coreExpression | Array<string | coreExpression>;
export type Expression = _Expression & {
  field?: string;
  originField?: string;
};
export type FilterWithDimension = {
  expr: Expression;
  name: string;
  dimension: Dimension;
};
export type Filters = {[key: string]: Filter};

///////////// WidgetConfig ColorItems  /////////////
export type ColorItem = {
  label: string;
  value: string;
  as: string;
  color: string;
};

///////////// WidgetConfig  /////////////
export type BaseWidgetConfig<D = Dimension, M = Measure> = {
  // widget id
  id: string;
  // widget type
  type: WIDGET | string;
  // widget source, it should be database table
  source: Source;
  // dimensions are the grouped columns in a query
  dimensions: D[];
  // measures are calculated fields such as SUM, AVERAGE
  measures: M[];
  // cross filter, key string
  filter: {
    [key: string]: Filter;
  };
  // self filter
  selfFilter: {
    [key: string]: Filter;
  };
  // widget layout
  layout: Layout;
  // widget title
  title?: string;
  // is this widget server rendered
  isServerRender?: boolean;
  //
  linkId?: string;
  // database sort
  sort?: {
    name: string;
    order: SORT_ORDER;
  };
  ignore?: string[];
  ignoreId?: any;
  // color items
  colorItems?: ColorItem[];
  // what color should we use
  colorKey?: string;
  // how many records are we query
  limit?: number;
  // where to start
  offset?: number;
  // ruler
  ruler?: any;
  // rulerbase
  rulerBase?: any;
  zoom?: number;
  // do we show the range chart
  isShowRange?: boolean;
  // other properties
  [propName: string]: any;
};

export type WidgetConfig = BaseWidgetConfig;
/////////////  Component Props  /////////////
export type MeasuresProps = {
  config: WidgetConfig;
  setConfig: Function;
  measuresSetting: MeasureSetting[];
  options: any[];
};

export type MeasureSelectorProps = {
  currOpt?: any;
  setting: MeasureSetting;
  placeholder?: string;
  measure?: Measure | undefined;
  options: any[];
  addMeasure: Function;
  deleteMeasure?: Function;
  enableAddColor?: boolean;
  [propName: string]: any;
};

export type ExpressionDropdownProps = {
  expressions: string[];
  measure: Measure;
  addMeasure: Function;
};

export type DimensionsProps = {
  config: WidgetConfig;
  setConfig: Function;
  dimensionsSetting: DimensionSetting[];
  options: any[];
};

export type DimensionSelectorProps = {
  id: string;
  source: Source;
  setting: DimensionSetting;
  placeholder?: string;
  dimension?: Dimension;
  options: any[];
  addDimension?: Function;
  deleteDimension?: Function;
  dLength?: number;
  enableAddColor?: boolean;
  [propName: string]: any;
};

export type BinProps = {
  id: string;
  source: Source;
  dimension: Dimension;
  addDimension: Function;
  staticRange?: Array<string | number>;
  onAdd?: Function;
  [propertyName: string]: any;
};

// chart related types
export type Mode = {
  mode: MODE;
  id: string;
};

export type WidgetProps<T> = {
  isLoading?: boolean;
  config: T;
  configs?: WidgetConfig[];
  setConfig: Function;
  mode: Mode;
  setMode: Function;
  data: Data;
  dataMeta: MetaInfo;
  linkData?: Data;
  linkMeta?: MetaInfo;
  wrapperWidth?: number;
  wrapperHeight?: number;
  chartHeight?: number;
  chartWidth?: number;
  chartHeightRatio?: number;
  dashboard: Dashboard;
};

export type DefaultWidgetProps = WidgetProps<WidgetConfig>;

export type HeaderProps = {
  showRestoreBtn?: boolean;
  title?: string;
  mode: Mode;
  setMode: Function;
  configs?: WidgetConfig[];
  config?: WidgetConfig;
  setConfig: Function;
  localConfig?: WidgetConfig;
  dashboard: Dashboard;
  setDashboard?: React.Dispatch<DashboardAction>;
  data?: DataCache;
  widgetSettings?: any;
};

export type SelectorProps = {
  options?: any[];
  currOpt?: any;
  onOptionChange: Function;
  onDelete?: any;
  [propName: string]: any;
};
