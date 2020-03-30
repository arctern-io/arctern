import {WidgetProps, BaseWidgetConfig} from '../../types';

export type PieChartConfig = BaseWidgetConfig & {
  sort: any;
  limit: number;
};export type PieChartProps = WidgetProps<PieChartConfig> & {
  onPieClick?: Function;
};