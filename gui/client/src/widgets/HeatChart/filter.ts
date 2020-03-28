import {cloneObj, id} from '../../utils/Helpers';
import {dimensionsDataToFilterExpr} from '../../utils/Filters';
import {WidgetConfig} from '../../types';

// HEAT Filters
export const heatCellFilterHandler = (config: any, cell: any) => {
  let copiedConfig = cloneObj(config);
  const isClearFilter = cell.data.filters.length > 0;
  copiedConfig.filter = copiedConfig.filter || {};

  if (isClearFilter) {
    // del filters
    cell.data.filters.forEach((f: any) => {
      delete copiedConfig.filter[f];
    });
  } else {
    copiedConfig.filter[id()] = {
      type: 'filter',
      expr: dimensionsDataToFilterExpr(cell.data.dimensionsData),
    };
  }

  return copiedConfig;
};

export const heatMultiFilterHandler = (config: WidgetConfig, multiDimensionsData: any) => {
  let copiedConfig = cloneObj(config);
  copiedConfig.filter = copiedConfig.filter || {};

  let filters = Object.keys(copiedConfig.filter);
  let filterExisting: boolean[] = [];

  multiDimensionsData.forEach((cell: any) => {
    const filterExpr: string = dimensionsDataToFilterExpr(cell);
    filters.forEach((f: string) => {
      let filter = copiedConfig.filter[f];
      let hasFilter = filter && filter.expr === filterExpr;
      if (hasFilter) {
        filterExisting.push(hasFilter);
        delete copiedConfig.filter[f];
      }
    });
  });

  const hasComboFilterExist = multiDimensionsData.length === filterExisting.length;
  const empty = filterExisting.length === 0;

  if (!hasComboFilterExist || empty) {
    multiDimensionsData.forEach((cell: any) => {
      const filterExpr: string = dimensionsDataToFilterExpr(cell);

      copiedConfig.filter[id()] = {
        type: 'filter',
        expr: filterExpr,
      };
    });
  }

  return copiedConfig;
};
