import {cloneObj} from '../../utils/Helpers';
import {
  dimensionGetter,
  dimensionTypeGetter,
  typeEqGetter,
  typeNormalizeStringGetter,
  dimensionBetweenExprGetter,
} from '../../utils/WidgetHelpers';

export const lineRangeFilterHandler = (config: any, range: any, rangeChart: boolean = false) => {
  let copiedConfig = cloneObj(config);
  const xDimension = dimensionGetter(config, 'x')!;
  const typeX = dimensionTypeGetter(xDimension);
  const eq = typeEqGetter(typeX);
  const normal = typeNormalizeStringGetter(typeX);
  let hasRange = !!copiedConfig.selfFilter.range;
  let isDeleting = range.length === 0;

  // if nothing changed, just do nothing
  if (
    hasRange &&
    eq(range[0], copiedConfig.selfFilter.range.expr.left) &&
    eq(range[1], copiedConfig.selfFilter.range.expr.right)
  ) {
    return false;
  }

  if (isDeleting) {
    delete copiedConfig.filter.range;
    delete copiedConfig.selfFilter.range;
    return copiedConfig;
  }

  // assign filter
  copiedConfig.filter.range = {
    type: 'filter',
    expr: dimensionBetweenExprGetter(xDimension, range.map(normal)),
  };
  if (rangeChart) {
    copiedConfig.selfFilter.range = {
      type: 'filter',
      expr: dimensionBetweenExprGetter(xDimension, range.map(normal)),
    };
  }

  return copiedConfig;
};

export const rangeDomainHandler = (config: any, xDomain: any) => {
  let copiedConfig = cloneObj(config);
  const xDimension = dimensionGetter(config, 'x')!;
  const typeX = dimensionTypeGetter(xDimension);
  const eq = typeEqGetter(typeX);
  const normal = typeNormalizeStringGetter(typeX);
  const hasXDomainSet = !!copiedConfig.selfFilter.xDomain;
  let isDeleting = xDomain.length === 0;
  // are we clearing domain
  const xDomainEq: boolean =
    hasXDomainSet &&
    eq(xDomain[0], copiedConfig.selfFilter.xDomain.expr.left) &&
    eq(xDomain[1], copiedConfig.selfFilter.xDomain.expr.right);

  if (xDomainEq) {
    return false;
  }

  let range = [normal(xDomain[0] || xDimension.min), normal(xDomain[1] || xDimension.max)];

  const filter = {
    type: 'filter',
    expr: dimensionBetweenExprGetter(xDimension, range),
  };
  copiedConfig.selfFilter.range = filter;
  copiedConfig.filter = {};
  if (isDeleting) {
    delete copiedConfig.selfFilter.range;
  }
  return copiedConfig;
};
