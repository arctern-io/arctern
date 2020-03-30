import {cloneObj} from '../../utils/Helpers';
import {measureGetter} from '../../utils/WidgetHelpers';
import {margin} from '../ScatterChart';
import {isGradientType, genColorGetter} from '../../utils/Colors';

export const rectHandler = ({config, width, height}: any) => {
  const copiedConfig = cloneObj(config);
  copiedConfig.width = Math.floor(width);
  copiedConfig.height = Math.floor(height);
  return copiedConfig;
};

export const getMarkerPos = ({data, config, x, y}: any) => {
  const xMeasure = measureGetter(config, 'x')!;
  const yMeasure = measureGetter(config, 'y')!;
  const {colorKey, colorItems = []} = config;

  const xValue = data[xMeasure.value];
  const yValue = data[yMeasure.value];
  const width = x(xValue);
  const height = y(yValue);
  const useColorItems = colorItems.length > 0;
  let color: any;
  if (useColorItems) {
    const targetKey = Object.keys(data).find((key: string) => key === colorItems[0].colName)!;
    color = colorItems.find(
      (item: any) => item.colName === targetKey && item.value === data[targetKey]
    )!.color;
  } else {
    color = isGradientType(colorKey) ? genColorGetter(config)(data.color) : colorKey;
  }
  return {x: width + margin.left, y: height + margin.top, color};
};
