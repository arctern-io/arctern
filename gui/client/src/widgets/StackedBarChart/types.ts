import {WidgetProps, BaseWidgetConfig} from '../../types';

export type StackedBarChartConfig = BaseWidgetConfig & {
  stackType: string;
}
export type StackedBarChartProps = WidgetProps<StackedBarChartConfig> & {
  onStackedBarClick: Function;
};
