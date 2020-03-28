
import { cloneObj, id } from "../../utils/Helpers";
import { dimensionGetter } from "../../utils/WidgetHelpers";

export const stackedBarFilterHander = (config: any, data: any) => {
  const copiedConfig = cloneObj(config);
  const { filter = {} } = copiedConfig;
  const xDimension = dimensionGetter(copiedConfig, 'x');
  const colName = xDimension && xDimension.value;
  const as = xDimension && xDimension.as;

  const value = data[as!];
  if (Object.keys(filter).length === 0) {
    filter[id()] = {
      type: "filter",
      expr: {
        type: "=",
        originField: colName,
        left: colName,
        right: value
      }
    };
  } else {
    const key = Object.keys(filter).find((key: string) => {
      const { expr } = filter[key];
      return expr.left === colName && expr.right === value;
    });
    key
      ? delete filter[key]
      : (filter[id()] = {
          type: "filter",
          expr: {
            type: "=",
            originField: colName,
            left: colName,
            right: value
          }
        });
  }
  copiedConfig.filter = filter;

  return copiedConfig;
};