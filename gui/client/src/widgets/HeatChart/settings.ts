import {makeSetting, defaultConfigHandler} from '../../utils/Setting';
import {orFilterGetter} from '../../utils/Filters';
import {onAddDimension, onDeleteDimension} from '../Utils/settingHelper';
import {COLUMN_TYPE, RequiredType} from '../../utils/Consts';

const heatChartConfigHandler = (config: any) => {
  let newConfig = defaultConfigHandler(config);
  newConfig.filter = orFilterGetter(config.filter);
  return newConfig;
};

const settings = makeSetting({
  type: 'HeatChart',
  icon: `<svg focusable="false" viewBox="0 0 48 48"><rect x="4" y="32" width="12" height="12"></rect><path d="M4,30h12V18H4V30z M7,21h6v6H7V21z"></path><path d="M4,16h12V4H4V16z M6,6h8v8H6V6z"></path><path d="M18,16h12V4H18V16z M20,6h8v8h-8V6z"></path><rect x="18" y="18" width="12" height="12"></rect><path d="M32,4v12h12V4H32z M41,13h-6V7h6V13z"></path><path d="M18,44h12V32H18V44z M21,35h6v6h-6V35z"></path><rect x="32" y="18" width="12" height="12"></rect><path d="M32,44h12V32H32V44z M34,34h8v8h-8V34z"></path></svg>`,
  enable: true,
  dimensions: [
    {
      type: RequiredType.REQUIRED,
      key: 'x',
      short: 'xaxis',
      columnTypes: [COLUMN_TYPE.NUMBER, COLUMN_TYPE.DATE, COLUMN_TYPE.TEXT],
      onAdd: onAddDimension,
      onDelete: onDeleteDimension,
    },
    {
      type: RequiredType.REQUIRED,
      key: 'y',
      short: 'yaxis',
      columnTypes: [COLUMN_TYPE.NUMBER, COLUMN_TYPE.DATE, COLUMN_TYPE.TEXT],
      onAdd: onAddDimension,
      onDelete: onDeleteDimension,
    },
  ],
  measures: [
    {
      type: RequiredType.REQUIRED,
      key: 'color',
      short: 'color',
      columnTypes: [COLUMN_TYPE.NUMBER, COLUMN_TYPE.TEXT],
    },
  ],
  configHandler: heatChartConfigHandler,
});

export default settings;
