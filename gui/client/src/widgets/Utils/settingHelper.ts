import {isTextCol} from '../../utils/ColTypes';
import {COLUMN_TYPE} from '../../utils/Consts';
import {cloneObj} from '../../utils/Helpers';
import {CONFIG} from '../../utils/Consts';
import {DEFAULT_SORT} from '../../components/settingComponents/Sort';
import {isRecordExist, QueryCount} from '../../utils/EditorHelper';
import {dimensionGetter} from '../../utils/WidgetHelpers';
import {color} from '../../utils/Colors';
import {toSQL} from '../../core/parser/reducer';
import {Dimension, Measure, ColorItem, WidgetConfig} from '../../types';
type BaseParams = {
  config: WidgetConfig; // or other WidgetSpecialConfig
  setConfig: Function;
  reqContext?: any;
};
export type MeasureParams = BaseParams & {
  measure: Measure;
};
export type DimensionParams = BaseParams & {
  dimension: Dimension;
};

type DistinctColumnParams = {
  config: WidgetConfig;
  setConfig: Function;
  dimension: Dimension | Measure;
};

export const prefixFilter = (prefix: string, name: string): string => {
  return `${prefix}_${name}`;
};

const _solveSortEffectsAfterDelete = ({
  config,
  target,
  setConfig,
}: {
  config: WidgetConfig;
  target: Dimension | Measure;
  setConfig: Function;
}) => {
  if (config.sort && config.sort.name === target.as) {
    setConfig({type: CONFIG.ADD_SORT, payload: DEFAULT_SORT});
  }
};

const _sortAfterDeleteDimension = ({config, dimension, setConfig}: DimensionParams) =>
  _solveSortEffectsAfterDelete({config, target: dimension, setConfig});

const _sortAfterDeleteMeasure = ({config, measure, setConfig}: MeasureParams) =>
  _solveSortEffectsAfterDelete({config, target: measure, setConfig});

// colorItems
const _colorItemsAfterAddMeasure = ({config, measure, setConfig}: MeasureParams) => {
  const colorDimension = dimensionGetter(config, 'color');
  if (!colorDimension) {
    return _addcolorItemsByMeasure({measure, setConfig});
  }
};

const _colorItemsAfterDeleteMeasure = ({config, measure, setConfig}: MeasureParams) => {
  const colorDimension = dimensionGetter(config, 'color');
  if (!colorDimension) {
    _deleteColorItemByMeasure({config, measure, setConfig});
  }
};

const _colorItemsAfterAddDimension = async ({
  config,
  dimension,
  setConfig,
  reqContext,
}: DimensionParams) => {
  const res = await queryDistinctValues({
    dimension,
    config,
    reqContext,
  });

  setConfig({type: CONFIG.DEL_COLORITEMS, payload: config.colorItems});
  cleanLastSelfFilter({dimension, setConfig, config});
  addSelfFilter({dimension, setConfig, res});
  const colorItems = parseTocolorItems(res);
  setConfig({type: CONFIG.ADD_COLORITEMS, payload: colorItems});
};

const _colorItemsAfterDeleteDimension = ({
  config,
  setConfig,
}: {
  config: WidgetConfig;
  setConfig: Function;
}) => {
  setConfig({type: CONFIG.DEL_COLORITEMS, payload: config.colorItems});
  const colorItems = config.measures.map((m: Measure) => genColorItem(m));
  setConfig({type: CONFIG.ADD_COLORITEMS, payload: colorItems});
};

// TODO:
const _resetSqlAfterChangeDimension = ({config}: {config: WidgetConfig}) => {
  const copyConfig = cloneObj(config);
  delete copyConfig.selfFilter.range;
  delete copyConfig.filter.range;
  return copyConfig;
};

// SelfFilter
const _selfFilterAfterDeleteDimension = ({dimension, setConfig, config}: DimensionParams) => {
  const key = prefixFilter('selfFilter', dimension.as);
  setConfig({type: CONFIG.DEL_SELF_FILTER, payload: {id: config.id, filterKeys: [key]}});
};

// Helpers
const _addcolorItemsByMeasure = ({measure, setConfig}: {measure: Measure; setConfig: Function}) => {
  const {label, as, isRecords, value} = measure;
  const newColorItem = {
    label,
    as,
    color: color(as),
    isRecords,
    value,
  };
  setConfig({type: CONFIG.ADD_COLORITEMS, payload: [newColorItem]});
};

const _deleteColorItemByMeasure = ({config, measure, setConfig}: MeasureParams) => {
  const targetColorItem = config.colorItems!.find((s: ColorItem) => s.as === measure.as);
  setConfig({type: CONFIG.DEL_COLORITEMS, payload: [targetColorItem]});
};

const _distinctSqlGetter = (
  dimension: Dimension | Measure,
  source: string,
  limit: number = QueryCount
) => {
  return toSQL({
    select: [`${dimension.value}`, `count (*) as countval`],
    from: source,
    groupby: [dimension.value],
    orderby: ['countval DESC'],
    limit: limit,
  });
};

export const cleanLastSelfFilter = ({dimension, setConfig, config}: DistinctColumnParams) => {
  const key = prefixFilter('selfFilter', dimension.as);
  setConfig({type: CONFIG.DEL_SELF_FILTER, payload: {id: config.id, filterKeys: [key]}});
};
export const addSelfFilter = ({
  dimension,
  res,
  setConfig,
}: {
  dimension: Dimension | Measure;
  setConfig: Function;
  res: {[key: string]: string}[];
}) => {
  const {value, as} = dimension;
  const key = prefixFilter('selfFilter', as);
  const filterVal = res.map((r: {[key: string]: string}) => r[value]);
  const targetSelfFilter = {
    [key]: {
      type: 'filter',
      expr: {
        type: 'in',
        set: filterVal,
        expr: value,
      },
    },
  };
  setConfig({type: CONFIG.ADD_SELF_FILTER, payload: targetSelfFilter});
};

export const onAddDimension = async ({
  config,
  dimension,
  setConfig,
  reqContext,
}: DimensionParams) => {
  cleanLastSelfFilter({dimension, setConfig, config});
  if (isTextCol(dimension.type)) {
    const res = await queryDistinctValues({config, dimension, reqContext});
    addSelfFilter({dimension, setConfig, res});
  }
  setConfig({type: CONFIG.ADD_DIMENSION, payload: {dimension}});
};

export const onDeleteDimension = ({config, dimension, setConfig}: DimensionParams) => {
  if (isTextCol(dimension.type)) {
    _selfFilterAfterDeleteDimension({config, dimension, setConfig});
  }
};

export const queryDistinctValues = async ({
  dimension,
  config,
  reqContext,
}: {
  dimension: Dimension | Measure;
  config: WidgetConfig;
  reqContext: any;
}) => {
  const sql = _distinctSqlGetter(dimension, config.source);
  const res = await reqContext.getTxtDistinctVal(sql);
  return res;
};

const _addDefaultMeasure = (config: WidgetConfig) => {
  config.measures.push({
    expression: 'count',
    value: '*',
    as: 'countval',
    format: 'auto',
    label: '',
    type: COLUMN_TYPE.NUMBER,
  });
  return config;
};

export const moveMeasureToDimension = (config: WidgetConfig) => {
  const cloneConfig = cloneObj(config);
  const {dimensions = [], measures = []} = cloneConfig;
  measures.forEach((measure: Measure) => {
    const {type = 'float8', value = '', as} = measure;
    dimensions.push({
      type,
      value,
      as,
    });
  });
  cloneConfig.dimensions = dimensions;
  cloneConfig.measures = [];
  return cloneConfig;
};

// Handlers
export const sortHandler = (config: WidgetConfig) => {
  let copiedConfig = cloneObj(config);
  if (!isRecordExist(copiedConfig)) {
    copiedConfig = _addDefaultMeasure(copiedConfig);
  }
  if (!copiedConfig.sort.name) {
    const recordMeasure = copiedConfig.measures.find((m: Measure) => m.isRecords);
    copiedConfig.sort.name = recordMeasure ? recordMeasure.as : 'countval';
  }
  return copiedConfig;
};

// SelfFilter

const Solver: any = {
  _sortAfterDeleteDimension,
  _sortAfterDeleteMeasure,
  _colorItemsAfterAddMeasure,
  _colorItemsAfterDeleteMeasure,
  _colorItemsAfterAddDimension,
  _colorItemsAfterDeleteDimension,
  // after delete or change dimension, we need to reset some configs.
  // delete configs: [filter.range,selfFilter.range]
  _resetSqlAfterChangeDimension,
};
export const solverGetter = (method: string, type: string, target: string) => {
  const keys = Object.keys(Solver);
  const [methodRegex, typeRegex, targetRegex] = [
    new RegExp(method, 'i'),
    new RegExp(type, 'i'),
    new RegExp(target, 'i'),
  ];
  const key = keys.find(
    (_key: string) => methodRegex.test(_key) && typeRegex.test(_key) && targetRegex.test(_key)
  );
  return key ? Solver[key] : () => {};
};

export const parseTocolorItems = (res: any[]) => {
  return res
    .filter(item => {
      const colName = Object.keys(item)[0];
      const value = item[colName];
      return value !== '';
    })
    .map((item: any) => {
      const colName = Object.keys(item)[0];
      const value = item[colName];
      return {
        colName,
        value,
        label: value,
        color: color(value),
        as: value,
      };
    });
};

export const genColorItem = (measure: Measure) => {
  const {value, as, label, isRecords} = measure;
  return {
    color: color(value),
    as,
    label,
    isRecords,
    value,
  };
};
