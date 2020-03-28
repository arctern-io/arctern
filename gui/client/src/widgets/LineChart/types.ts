import {WidgetProps, BaseWidgetConfig} from '../../types';

export type LineChartConfig = BaseWidgetConfig & {
  colorItems?: any[];
  isShowRange?: boolean;
  ruler?: any;
  isArea: boolean;
  filter: {
    range: any;
    [key: string]: any;
  };
};export type LineChartProps = WidgetProps<LineChartConfig> & {
  setConfig: Function;
  onRangeChange?: Function;
  onDomainChange?: Function;
  showXLabel?: boolean;
  isRange?: boolean;
};