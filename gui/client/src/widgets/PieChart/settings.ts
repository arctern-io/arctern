import {makeSetting} from '../../utils/Setting';
import {orFilterGetter} from '../../utils/Filters';
import {onAddDimension, solverGetter, sortHandler} from '../Utils/settingHelper';
import {COLUMN_TYPE, RequiredType} from '../../utils/Consts';

const onDeleteBarPieDimension = ({config, dimension, setConfig}: any) => {
  const sortSolver = solverGetter('delete', 'dimension', 'sort');
  return sortSolver({config, dimension, setConfig});
};

const barPieConfigHandler = (config: any) => {
  let newConfig = sortHandler(config);
  newConfig.filter = orFilterGetter(config.filter);
  return newConfig;
};

const settings = makeSetting({
  type: 'PieChart',
  dimensions: [
    {
      type: RequiredType.REQUIRED_ONE_AT_LEAST,
      columnTypes: [COLUMN_TYPE.NUMBER, COLUMN_TYPE.DATE, COLUMN_TYPE.TEXT],
      onAdd: onAddDimension,
      onDelete: onDeleteBarPieDimension,
    },
  ],
  measures: [
    {
      type: RequiredType.REQUIRED,
      key: 'size',
      short: 'size',
      columnTypes: [COLUMN_TYPE.NUMBER, COLUMN_TYPE.TEXT],
    },
  ],
  icon: `<svg   focusable="false" viewBox="0 0 48 48" aria-hidden="true" role="presentation"><path d="M40.6,12.9l3.3-2.2C39.7,4.2,32.3,0,24,0v4C13,4,4,13,4,24s9,20,20,20s20-9,20-20C44,19.9,42.8,16.1,40.6,12.9z M26.5,2.6 c5.4,0.6,10.4,3.3,13.9,7.4l-13.9,9.3V2.6z M24,40.5c-9.1,0-16.5-7.4-16.5-16.5S14.9,7.5,24,7.5V24l13.7-9.1 c1.8,2.6,2.8,5.8,2.8,9.1C40.5,33.1,33.1,40.5,24,40.5z"></path></g></svg>`,
  enable: true,
  configHandler: barPieConfigHandler,
});

export default settings;
