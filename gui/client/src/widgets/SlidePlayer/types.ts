import {BaseWidgetConfig} from '../../types';

export type SlidePlayerConfig = BaseWidgetConfig & {
  filter: {
    range: any;
    [key: string]: any;
  };
};

export type SlidePlayerProps = {
  wrapperWidth: number;
  wrapperHeight: number;
  config: SlidePlayerConfig;
  dataMeta: any;
  onRangeChange: Function;
  data: dateType[];
  setConfig?: Function;
};

type dateType = {
  value: string;
  as: string;
  options: [];
};
