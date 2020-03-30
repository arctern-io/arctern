import {orFilterGetter} from '../../utils/Filters';
import {makeSetting} from '../../utils/Setting';
import {onAddDimension, solverGetter, sortHandler} from '../Utils/settingHelper';
import {RequiredType, COLUMN_TYPE} from '../../utils/Consts';

export const onDeleteBarPieDimension = ({config, dimension, setConfig}: any) => {
  const sortSolver = solverGetter('delete', 'dimension', 'sort');
  return sortSolver({config, dimension, setConfig});
};

const barPieConfigHandler = (config: any) => {
  let newConfig = sortHandler(config);
  newConfig.filter = orFilterGetter(config.filter);
  return newConfig;
};

const settings = makeSetting({
  dimensions: [
    {
      type: RequiredType.REQUIRED_ONE_AT_LEAST,
      columnTypes: [COLUMN_TYPE.DATE, COLUMN_TYPE.NUMBER, COLUMN_TYPE.TEXT],
      onAdd: onAddDimension,
      onDelete: onDeleteBarPieDimension,
    },
  ],
  measures: [
    {
      type: RequiredType.REQUIRED,
      key: 'width',
      short: 'width',
      columnTypes: [COLUMN_TYPE.NUMBER, COLUMN_TYPE.TEXT],
    },
    {
      type: RequiredType.OPTION,
      key: 'color',
      short: 'color',
      columnTypes: [COLUMN_TYPE.NUMBER, COLUMN_TYPE.TEXT],
    },
  ],
  type: 'BarChart',
  icon: `<svg focusable="false" viewBox="0 0 48 48" aria-hidden="true" role="presentation"><g id="icon-chart-row"><rect x="4" y="4" width="2" height="40"></rect><rect x="10" y="16" width="26" height="6"></rect><rect x="10" y="6" width="34" height="6"></rect><rect x="10" y="26" width="18" height="6"></rect><rect x="10" y="36" width="14" height="6"></rect></g></svg>`,
  enable: true,
  configHandler: barPieConfigHandler,
});

export default settings;
