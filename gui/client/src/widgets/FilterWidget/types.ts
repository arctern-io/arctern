import {BaseWidgetConfig} from '../../types';

type FilterWidgetConfig = BaseWidgetConfig;

export type FilterWidgetViewProps = {
  config: FilterWidgetConfig;
  setConfig: Function;
  wrapperWidth: number;
  wrapperHeight: number;
};

export type FilterWidgetProps = {
  items: Item[];
  wrapperWidth: number;
  wrapperHeight: number;
  onClick: Function;
};

export type Item = {
  options: Array<Option>;
  as: string;
  value: string;
};

export type Option = {
  label: string;
  value: Value;
  isSelected: boolean;
};

export type Value = number | string | Array<Date | number>;
