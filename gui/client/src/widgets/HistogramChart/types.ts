import {WidgetProps, BaseWidgetConfig} from '../../types';


export type HistogramChartConfig = BaseWidgetConfig & {
  colorItems?: any[];
  isShowRange?: boolean;
  ruler?: any;
  filter: {
    range: any;
    [key: string]: any;
  };
};export type HistogramChartProps = WidgetProps<HistogramChartConfig> & {
  setConfig: Function;
  onRangeChange?: Function;
  isRange?: boolean;
  showXLabel?: boolean;
};