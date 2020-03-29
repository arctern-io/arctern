import {WidgetProps, BaseWidgetConfig} from '../../types';

export type HeatChartConfig = BaseWidgetConfig;
export type HeatChartProps = WidgetProps<HeatChartConfig> & {
  onCellClick?: Function;
  onRowClick?: Function;
  onColClick?: Function;
};
