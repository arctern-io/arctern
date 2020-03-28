import {cloneObj} from './Helpers';
import {WidgetConfig, InitSetting, Setting, ConfigHandler, InitMeasureSetting} from '../types';
import {COLUMN_TYPE, EXPRESSION_OPTIONS, DefaultExpressionOption, RequiredType} from './Consts';
export const defaultConfigHandler: ConfigHandler = config => cloneObj(config);

const defaultOnAfterSqlCreate = (sql: string): string => sql;
const DEFAULT_COLUMN_TYPES = [COLUMN_TYPE.NUMBER, COLUMN_TYPE.DATE, COLUMN_TYPE.TEXT];
export function makeSetting<T extends WidgetConfig>(setting: InitSetting<T>): Setting<T> {
  if (setting.dimensions.length === 0) {
    setting.dimensions = [{type: RequiredType.NONE_REQUIRED, columnTypes: DEFAULT_COLUMN_TYPES}];
  }
  if (setting.measures.length === 0) {
    setting.measures = [{type: RequiredType.NONE_REQUIRED, columnTypes: DEFAULT_COLUMN_TYPES}];
  }
  if (typeof setting.configHandler !== 'function') {
    setting.configHandler = config => cloneObj(config);
  }

  if (typeof setting.onAfterSqlCreate !== 'function') {
    setting.onAfterSqlCreate = defaultOnAfterSqlCreate;
  }

  if (typeof setting.enable === 'undefined') {
    setting.enable = true;
  }

  if (typeof setting.isServerRender === 'undefined') {
    setting.isServerRender = false;
  }
  setting.measures = setting.measures.map((m: InitMeasureSetting) => {
    if (!m.expressions) {
      m.expressions = EXPRESSION_OPTIONS.map((opt: DefaultExpressionOption) => opt.value);
    }
    if (m.expressions.length === 0) {
      m.expressions = ['project'];
    }
    return m;
  });

  return setting as Setting<T>;
}
