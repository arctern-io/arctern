import {WidgetProps, BaseWidgetConfig} from '../../types';
export type ScatterChartConfig = BaseWidgetConfig 

export type ScatterChartProps = WidgetProps<ScatterChartConfig> & {
  onZooming?: Function;
  onZoomEnd?: Function;
  onMouseMove?: Function;
  onMouseLeave?: Function;
  onZoorefmEnd?: Function;
  onRectChange?: Function;
  reset?: Function;
  radius?: number;
};
