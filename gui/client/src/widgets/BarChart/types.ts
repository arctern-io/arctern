import {WidgetProps, BaseWidgetConfig, Data as _Data} from '../../types';

export type Data = _Data;
export type BarWidgetConfig = BaseWidgetConfig;
export type BarChartProps = WidgetProps<BaseWidgetConfig> & {
  onBarClick?: Function;
};
