import {id as genID} from '../../../utils/Helpers';
import {prefixFilter} from '../../Utils/settingHelper';
import {CONFIG} from '../../../utils/Consts';
import {dimensionGetter} from '../../../utils/WidgetHelpers';
import {Filter, BaseWidgetConfig, WidgetConfig} from '../../../types';
import {getColType} from '../../../utils/ColTypes';
import {getNumSeconds, TimeBin} from '../../../utils/Time';

export const genFilter = (val: any, target: any) => {
  const {isBinned, timeBin = '', extract = false, maxbins, value} = target;
  const isTimeBinDimension = timeBin && !extract;
  const isNumBinDimension = !!maxbins;
  const isMeasureOrTextDimension = !isBinned;
  let newFilter: any;
  if (isNumBinDimension) {
    newFilter = {
      type: 'filter',
      expr: {
        type: 'between',
        field: value,
        left: val[0],
        right: val[1],
      },
    };
  }
  if (isTimeBinDimension) {
    newFilter = {
      type: 'filter',
      expr: {
        type: 'between',
        field: value,
        originField: value,
        left: val[0],
        right: val[1],
      },
    };
  }
  if (extract) {
    newFilter = {
      type: 'filter',
      expr: {
        type: 'between',
        originField: value,
        field: `extract('${timeBin}' from ${value})`,
        left: val[0],
        right: val[1],
      },
    };
  }
  if (isMeasureOrTextDimension) {
    newFilter = {
      type: 'filter',
      expr: {
        type: '=',
        originField: value,
        left: value,
        right: val,
      },
    };
  }
  return newFilter;
};

export const findExistFilterKey = (newFilter: any, filter: any) => {
  return Object.keys(filter).find(
    (key: any) => JSON.stringify(filter[key]) === JSON.stringify(newFilter)
  );
};

export const checkIsSelected = (value: any, target: any, filter: any = {}) => {
  const fakeFilter = genFilter(value, target);
  return !!findExistFilterKey(fakeFilter, filter);
};

export const addTextSelfFilter = ({val, key, config, setConfig}: any) => {
  const targetKey = prefixFilter('selfFilter', key);
  const targetSelfFilter = config['selfFilter'][targetKey];
  targetSelfFilter.expr.set.push(val);
  setConfig({type: CONFIG.ADD_SELF_FILTER, payload: {[targetKey]: targetSelfFilter}});
};

type BasicFilterParams = {
  as: string;
  config: BaseWidgetConfig;
  setConfig: Function;
};

type TextFilterParams = BasicFilterParams & {
  val: string;
};
type NumberFilterParams = BasicFilterParams & {
  val: number;
};
type DateFilterParams = BinningFilterParams | ExtractFilterParams;
type BinningFilterParams = BasicFilterParams & {
  val: Date | string;
};
type ExtractFilterParams = BasicFilterParams & {
  val: number;
};

type FilterParams =
  | TextFilterParams
  | NumberFilterParams
  | BinningFilterParams
  | ExtractFilterParams;

const _isMatchedFilterKey = (as: string, filterKey: string) => {
  const regex = new RegExp(as);
  return !!regex.test(filterKey);
};
const getMatchedFilterKeys = (as: string = '', filter: Filter) => {
  return Object.keys(filter).filter((key: string) => _isMatchedFilterKey(as, key));
};
const _addTextFilter = ({val, as, config, setConfig}: TextFilterParams) => {
  const dimension = dimensionGetter(config, as)!;
  const targetFilter = config.filter[as] || {
    type: 'filter',
    expr: {type: 'in', set: [], expr: dimension.value},
  };
  targetFilter.expr.set.push(val);
  setConfig({type: CONFIG.ADD_FILTER, payload: {[as]: targetFilter, id: config.id}});
};
const _deleteTextFilter = ({val, as, config, setConfig}: TextFilterParams) => {
  let filter = config.filter[as];
  filter.expr.set = filter.expr.set.filter((item: string) => item !== val);
  filter.expr.set.length === 0
    ? setConfig({type: CONFIG.DEL_FILTER, payload: {filterKeys: [as], id: config.id}})
    : setConfig({type: CONFIG.ADD_FILTER, payload: {[as]: filter, id: config.id}});
};
const _addNumberFilter = ({val, as, config, setConfig}: NumberFilterParams) => {
  const filterKey = prefixFilter(as, genID());
  const dimension = dimensionGetter(config, as)!;
  const range = _rangeGetter(val, as, config);
  const newFilter = {
    [filterKey]: {
      type: 'filter',
      expr: {
        type: 'between',
        field: dimension.value,
        left: range[0],
        right: range[1],
      },
    },
  };
  setConfig({type: CONFIG.ADD_FILTER, payload: {...newFilter, id: config.id}});
};
const _deleteNumberFilter = ({val, as, config, setConfig}: NumberFilterParams) => {
  // @ts-ignore
  const matchedKeys = getMatchedFilterKeys(as, config.filter);
  if (matchedKeys.length === 0) {
    return false;
  }
  const range = _rangeGetter(val, as, config);
  const targetKey = matchedKeys.find((key: string) => {
    const {left, right} = config.filter[key].expr;
    return left === range[0] && right === range[1];
  });
  if (!targetKey) {
    return false;
  }
  setConfig({type: CONFIG.DEL_FILTER, payload: {id: config.id, filterKeys: [targetKey]}});
};
const _addBinningDateFilter = ({val, as, config, setConfig}: BinningFilterParams) => {
  const filterKey = prefixFilter(as, genID());
  const dimension = dimensionGetter(config, as)!;
  const range = _rangeGetter(val as string, as, config);
  const newFilter = {
    [filterKey]: {
      type: 'filter',
      expr: {
        type: 'between',
        field: dimension.value,
        left: (range[0] as Date).toUTCString(),
        right: (range[1] as Date).toUTCString(),
      },
    },
  };
  setConfig({type: CONFIG.ADD_FILTER, payload: {...newFilter, id: config.id}});
};
const _addExtractDateFilter = ({val, as, config, setConfig}: ExtractFilterParams) => {
  const filterKey = prefixFilter(as, genID());
  const dimension = dimensionGetter(config, as)!;
  const newFilter = {
    [filterKey]: {
      type: 'filter',
      expr: {
        type: 'between',
        originField: dimension.value,
        field: `extract('${dimension.timeBin}' from ${dimension.value})`,
        left: val + 1,
        right: val + 2,
      },
    },
  };
  setConfig({type: CONFIG.ADD_FILTER, payload: {...newFilter, id: config.id}});
};
const _deleteDateFilter = (param: DateFilterParams) => {
  const dimension = dimensionGetter(param.config, param.as)!;
  dimension.extract
    ? _deleteExtractFilter(param as ExtractFilterParams)
    : _deleteBinningFilter(param as BinningFilterParams);
};
const _deleteBinningFilter = ({val, as, config, setConfig}: BinningFilterParams) => {
  // @ts-ignore
  const matchedKeys = getMatchedFilterKeys(as, config.filter);
  if (matchedKeys.length === 0) {
    return false;
  }
  const targetKey = matchedKeys.find((key: string) => {
    const {left, right} = config.filter[key].expr;
    const range = _rangeGetter(val as string, as, config);
    return left === (range[0] as Date).toUTCString() && right === (range[1] as Date).toUTCString();
  });
  if (!targetKey) {
    return false;
  }
  setConfig({type: CONFIG.DEL_FILTER, payload: {id: config.id, filterKeys: [targetKey]}});
};
const _deleteExtractFilter = ({val, as, config, setConfig}: ExtractFilterParams) => {
  const dimension = dimensionGetter(config, as)!;
  // @ts-ignore
  const matchedKeys = getMatchedFilterKeys(as, config.filter);
  let targetKey = matchedKeys.find((key: string) => {
    const {originField, left, right} = config.filter[key].expr;
    return originField === dimension.value && left === val + 1 && right === val + 2;
  });
  setConfig({type: CONFIG.DEL_FILTER, payload: {filterKeys: [targetKey], id: config.id}});
};

const _rangeGetter = (
  val: number | string,
  as: string,
  config: WidgetConfig
): string[] | number[] | Date[] => {
  const dimension = dimensionGetter(config, as)!;
  const {maxbins = 2, extent = [], timeBin, extract} = dimension;
  const isDateBin = !!timeBin && !extract;
  const isNumberBin = !!maxbins;
  let min, max;
  if (isDateBin) {
    const numSeconds = getNumSeconds(timeBin as TimeBin);
    min = new Date(val as string).getTime();
    max = min + numSeconds * 1000;
    return [new Date(min), new Date(max)];
  }
  if (isNumberBin) {
    const stepRange = ((extent[1] as number) - (extent[0] as number)) / maxbins;
    return [val as number, (val as number) + stepRange];
  }
  return [];
};
const _addFilter = ({val, as, config, setConfig}: FilterParams) => {
  switch (_getType(as, config)) {
    case 'text':
      _addTextFilter({val: val as string, as, config, setConfig});
      break;
    case 'number':
      _addNumberFilter({val: val as number, as, config, setConfig});
      break;
    case 'date':
      const dimension = dimensionGetter(config, as)!;
      dimension.extract
        ? _addExtractDateFilter({val: val as number, as, config, setConfig})
        : _addBinningDateFilter({val: val as string, as, config, setConfig});
      break;
    default:
      break;
  }
};
const _deleteFilter = ({val, as, config, setConfig}: FilterParams) => {
  switch (_getType(as, config)) {
    case 'text':
      _deleteTextFilter({val: val as string, as, config, setConfig});
      break;
    case 'number':
      _deleteNumberFilter({val: val as number, as, config, setConfig});
      break;
    case 'date':
      const dimension = dimensionGetter(config, as)!;
      dimension.extract
        ? _deleteExtractFilter({val: val as number, as, config, setConfig})
        : _deleteDateFilter({val: val as number, as, config, setConfig});
      break;
    default:
      break;
  }
};
const _getType = (as: string, config: WidgetConfig) => {
  const dimension = dimensionGetter(config, as)!;
  return getColType(dimension.type);
};
const _isFilterValid = (as: string, config: WidgetConfig): Boolean => {
  const dimension = dimensionGetter(config, as);
  let isFilterExist = false;
  switch (_getType(as, config)) {
    case 'text':
      isFilterExist = Object.keys(config.filter).some((key: string) => key === as);
      return !!(dimension && isFilterExist);
    case 'number':
    case 'date':
      isFilterExist = !!Object.keys(config.filter).find((key: string) => {
        const regex = new RegExp(as);
        return regex.test(key);
      });
      return !!(dimension && isFilterExist);
    default:
      return false;
  }
};
const _isTextTypeSelected = (val: string, as: string, config: WidgetConfig): Boolean => {
  return (
    _isFilterValid(as, config) &&
    config.filter[as].expr &&
    config.filter[as].expr.set.some((opt: string) => opt === val)
  );
};
const _isNumberTypeSelected = (val: number, as: string, config: WidgetConfig): Boolean => {
  const dimension = dimensionGetter(config, as)!;
  const [min, max] = _rangeGetter(val, as, config);
  const isSelected = Object.keys(config.filter).some((key: string) => {
    const filter = config.filter[key];
    return (
      filter.expr &&
      filter.expr.field === dimension.value &&
      filter.expr.left === min &&
      filter.expr.right === max
    );
  });
  return _isFilterValid(as, config) && !!isSelected;
};
const _isBinningSelected = (val: Date | string, as: string, config: WidgetConfig) => {
  const dimension = dimensionGetter(config, as)!;
  const [min, max] = _rangeGetter(val as string, as, config);

  const isSelected = Object.keys(config.filter).some((key: string) => {
    const filter = config.filter[key];
    return (
      filter.expr &&
      filter.expr.field === dimension.value &&
      filter.expr.left === new Date(min).toUTCString() &&
      filter.expr.right === new Date(max).toUTCString()
    );
  });
  return _isFilterValid(as, config) && !!isSelected;
};
const _isExtractSelected = (val: number, as: string, config: WidgetConfig) => {
  const {value, timeBin} = dimensionGetter(config, as) || {};
  const isSelected = Object.keys(config.filter).some((key: string) => {
    const filter = config.filter[key];
    return (
      filter.expr &&
      filter.expr.originField === value &&
      filter.expr.field === `extract('${timeBin}' from ${value})` &&
      filter.expr.left === val + 1 &&
      filter.expr.right === val + 2
    );
  });
  return _isFilterValid(as, config) && isSelected;
};
const _isDateTypeSelected = (val: string | number, as: string, config: WidgetConfig) => {
  const dimension = dimensionGetter(config, as)!;
  return dimension.extract
    ? _isExtractSelected(val as number, as, config)
    : _isBinningSelected(val as string, as, config);
};
export const isSelected = (val: any, as: string, config: WidgetConfig) => {
  switch (_getType(as, config)) {
    case 'text':
      return _isTextTypeSelected(val, as, config);
    case 'number':
      return _isNumberTypeSelected(val, as, config);
    case 'date':
      return _isDateTypeSelected(val, as, config);
    default:
      return false;
  }
};

// when filter's key is related to dimension's as(like FilterWidget and TableWidget), use this filterHandler
export const handleFilter = ({val, as, config, setConfig}: FilterParams) => {
  isSelected(val, as, config)
    ? _deleteFilter({val, as, config, setConfig})
    : _addFilter({val, as, config, setConfig});
};
