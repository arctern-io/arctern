import {makeSetting} from '../../utils/Setting';
import {cloneObj} from '../../utils/Helpers';
import {orFilterGetter} from '../../utils/Filters';
import {sortHandler, moveMeasureToDimension, solverGetter} from '../Utils/settingHelper';
import {RequiredType, COLUMN_TYPE} from '../../utils/Consts';

// TableChart
const onDeleteTableChartDimension = ({config, dimension, setConfig}: any) => {
  const sortSolver = solverGetter('delete', 'dimension', 'sort');
  sortSolver({config, dimension, setConfig});
};
const onDeleteTableChartMeasure = ({config, measure, setConfig}: any) => {
  const sortSolver = solverGetter('delete', 'measure', 'sort');
  sortSolver({config, measure, setConfig});
};

const tableConfigHandler = (config: any) => {
  let newConfig = cloneObj(config);
  newConfig.filter = orFilterGetter(newConfig.filter);

  if (newConfig.dimensions.length === 0) {
    newConfig = moveMeasureToDimension(newConfig);
  }

  newConfig = sortHandler(newConfig);
  return newConfig;
};

const settings = makeSetting({
  type: 'TableChart',
  icon: `<svg focusable="false" viewBox="0 0 48 48" aria-hidden="true" role="presentation"><g id="icon-chart-table"><path d="M4,44h40V4H4V44z M41,41H7V10h34V41z"></path><path d="M8,34v2h7v4h2v-4h14v4h2v-4h7v-2h-7v-6h7v-2h-7v-6h7v-2h-7v-5h-2v5H17v-5h-2v5H8v2h7v6H8v2h7v6H8z M17,20h14v6H17V20z M17,28h14v6H17V28z"></path></g></svg>`,
  dimensions: [
    {
      type: RequiredType.REQUIRED_ONE_DIMENSION_OR_MEASURE_AT_LEAST,
      short: 'colname',
      columnTypes: [COLUMN_TYPE.DATE, COLUMN_TYPE.NUMBER, COLUMN_TYPE.TEXT],
      isNotUseBin: false,
      onDelete: onDeleteTableChartDimension,
    },
  ],
  measures: [
    {
      type: RequiredType.REQUIRED_ONE_DIMENSION_OR_MEASURE_AT_LEAST,
      short: 'colname',
      columnTypes: [COLUMN_TYPE.DATE, COLUMN_TYPE.NUMBER, COLUMN_TYPE.TEXT],
      onDelete: onDeleteTableChartMeasure,
    },
  ],
  enable: true,
  configHandler: tableConfigHandler,
});

export default settings;
