import { cloneObj, id } from "../../utils/Helpers";
import { dimensionsDataToFilterExpr } from "../../utils/Filters";

export const pieFilterHandler = (config: any, pie: any) => {
  let copiedConfig = cloneObj(config);
  const isClearFilter = pie.data.filters.length > 0;
  copiedConfig.filter = copiedConfig.filter || {};

  if (isClearFilter) {
    // del filters
    pie.data.filters.forEach((f: any) => {
      delete copiedConfig.filter[f];
    });
  } else {
    copiedConfig.filter[id()] = {
      type: "filter",
      expr: dimensionsDataToFilterExpr(pie.data.dimensionsData)
    };
  }
  return copiedConfig;
};
