import {cloneObj, id} from '../../utils/Helpers';
import {dimensionsDataToFilterExpr} from '../../utils/Filters';

export const barFilterHandler = (config: any, bar: any) => {
  let copiedConfig = cloneObj(config);
  const isClearFilter = bar.data.filters.length > 0;
  copiedConfig.filter = copiedConfig.filter || {};

  if (isClearFilter) {
    // del filters
    bar.data.filters.forEach((f: any) => {
      delete copiedConfig.filter[f];
    });
  } else {
    copiedConfig.filter[id()] = {
      type: 'filter',
      expr: dimensionsDataToFilterExpr(bar.data.dimensionsData),
    };
  }
  return copiedConfig;
};
