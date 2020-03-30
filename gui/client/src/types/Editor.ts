import {WidgetConfig, Setting, Dashboard} from '.';

export type WidgetEditorProps = {
  config: WidgetConfig;
  setConfig: Function;
  setting: Setting;
  dashboard: Dashboard;
  [propName: string]: any;
};

// Status of DimensionSelector or MeasureSelector
export enum Status {
  SELECTED = 'selected',
  SELECT_COLUMN = 'selectColumn',
  CUSTOM = 'custom',
  ADD = 'add',
  // Only in DimensionSelector
  SELECT_BIN = 'selectBin',
  // Only in MeasureSelector
  SELECT_EXPRESSION = 'selectExpression',
}

export type WidgetSelectorProps = {
  icon: string;
  widgetType: string;
  selected: boolean;
  onClick: Function;
};

export type CustomSqlInputProp = {
  currVal: {
    label: string;
    value: string;
  };
  placeholder: {
    name: string;
    funcText: string;
  };
  onSave: Function;
  onCancel: Function;
};

export type CustomSqlOptProps = {
  placeholder: string;
  onClick: Function;
};
