import {cloneObj, id} from '../../utils/Helpers';
import {dimensionGetter} from '../../utils/WidgetHelpers';
import {WidgetConfig} from '../../types';
import {findExistFilterKey} from '../Utils/filters/common';
import {getNumSeconds} from '../../utils/Time';
import {dimensionsDataToFilterExpr} from '../../utils/Filters';

const getRange = (dimension: any, value: any) => {
  const {isBinned, timeBin, extract = false, maxbins, extent = []} = dimension,
    [min, max] = extent,
    isTimeBinDimension = timeBin && !extract,
    isNumBinDimension = !!maxbins,
    isMeasureOrTextDimension = !isBinned && !extract;
  let filterValue: any;
  if (isNumBinDimension) {
    let gap = (max - min) / maxbins,
      _min = min + gap * value,
      _max = _min + gap;
    filterValue = [_min, _max];
  }
  if (isTimeBinDimension) {
    let _max = new Date(
      new Date(value).getTime() + getNumSeconds(timeBin) * 1000 - 1
    ).toUTCString();
    filterValue = [value, _max];
  }
  if (extract) {
    filterValue = [value, value + 1];
  }
  if (isMeasureOrTextDimension) {
    filterValue = [value];
  }
  return filterValue;
};

const dimensionDataGetter = (item: any = {}, config: WidgetConfig) => {
  return Object.keys(item)
    .filter((key: string) => !!dimensionGetter(config, key))
    .map((key: string) => {
      const dimension = dimensionGetter(config, key);
      const data = getRange(dimension, item[key]);
      return {dimension, data};
    });
};
export const bubbleFilterHandler = (config: WidgetConfig, item: any) => {
  const copiedConfig = cloneObj(config),
    {filter = {}} = copiedConfig;
  const dimensionData = dimensionDataGetter(item, config);
  const newFilter = {
    type: 'filter',
    expr: dimensionsDataToFilterExpr(dimensionData),
  };
  const isExistKey = findExistFilterKey(newFilter, filter);
  isExistKey ? delete filter[isExistKey] : (filter[id()] = newFilter);
  copiedConfig.filter = filter;
  return copiedConfig;
};

export const checkIsSelected = (item: any, config: any) => {
  const {filter = {}} = config;
  const keys = Object.keys(filter);
  if (keys.length === 0) {
    return true;
  }
  const dimensionData = dimensionDataGetter(item, config);
  const newFilter = {
    type: 'filter',
    expr: dimensionsDataToFilterExpr(dimensionData),
  };
  return !!findExistFilterKey(newFilter, filter);
};
