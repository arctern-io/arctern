import {dimensionBetweenExprGetter, dimensionEqExprGetter} from './WidgetHelpers';
import {WidgetConfig, Filters, Expression, Dimension, FilterWithDimension} from '../types';
import {SqlParser} from 'infinivis-core';
const {parseExpression} = SqlParser;
export const widgetFiltersGetter = (config: WidgetConfig) => {
  let filters: Array<FilterWithDimension | undefined> = Object.keys(config.filter || {})
    .map(f => {
      const dimension = config.dimensions.filter((d: Dimension) => {
        const expression: Expression = config.filter[f].expr;
        return d.value === expression.field || d.value === expression.originField;
      })[0];

      return dimension
        ? {
            expr: config.filter[f].expr,
            name: f,
            dimension: dimension,
          }
        : undefined;
    })
    .filter(f => f);
  return filters;
};

export const orFilterGetter = (filters: Filters = {}, filterId: string = 'orFilter') => {
  let newFilters: any = {};
  let or: any = [];
  Object.keys(filters).forEach((f: string) => {
    or.push(parseExpression(filters[f].expr as string));
  });
  if (or.length) {
    newFilters[filterId] = {
      type: 'filter',
      expr: `(${or.join(`) OR (`)})`,
    };
  }
  return newFilters;
};

export const andFilterGetter = (filters: Filters = {}, filterId: string = 'andFilter') => {
  let newFilters: Filters = {};
  let and: Array<string | Expression> = [];
  Object.keys(filters).forEach((f: any) => {
    and.push(parseExpression(filters[f].expr as string));
  });

  if (and.length) {
    newFilters[filterId] = {
      type: 'filter',
      expr: and.join(` AND `),
    };
  }
  return newFilters;
};

export const dimensionsDataToFilterExpr = (dimensionsData: any[]) => {
  const filters: Filters = {};
  // add filters
  dimensionsData.forEach((dimensionData: any, index: number) => {
    const {timeBin, isBinned, extract} = dimensionData.dimension;
    const isTimeBin = timeBin && !extract;
    const isNumberBin = isBinned && !extract;
    // bin filter
    if (isTimeBin || isNumberBin) {
      filters[index] = {
        type: 'filter',
        expr: dimensionBetweenExprGetter(dimensionData.dimension, dimensionData.data),
      };
      return;
    }
    // normal eq filter
    filters[index] = {
      type: 'filter',
      expr: dimensionEqExprGetter(dimensionData.dimension, dimensionData.data),
    };
  });
  return andFilterGetter(filters).andFilter.expr;
};

export const getFilterLength = (filters: Filters[]): number => {
  let count = 0;
  filters.forEach(f => {
    count += Object.keys(f).length;
  });
  return count;
};
