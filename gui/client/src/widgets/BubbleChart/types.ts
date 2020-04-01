import {WidgetProps, BaseWidgetConfig} from '../../types';

export type BubbleChartConfig = BaseWidgetConfig;
export type BubbleChartProps = WidgetProps<BubbleChartConfig> & {
  onClick?: Function;
};
