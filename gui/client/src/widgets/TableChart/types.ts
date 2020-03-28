import {WidgetProps, BaseWidgetConfig} from '../../types';

export type TableChartConfig = BaseWidgetConfig & {
  offset: number;
  length: number;
};
export type TableChartProps = WidgetProps<TableChartConfig> & {
  onColumnClick: Function;
  onSortChange: Function;
  onBottomReached?: Function;
};
