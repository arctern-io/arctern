import {makeSetting} from '../../utils/Setting';
import {orFilterGetter} from '../../utils/Filters';
import {WidgetConfig} from '../../types';
import {
  DimensionParams,
  MeasureParams,
  onAddDimension,
  onDeleteDimension,
  solverGetter,
  sortHandler,
} from '../Utils/settingHelper';
import {CONFIG, RequiredType, COLUMN_TYPE} from '../../utils/Consts';

const onDeleteXDimension = ({config, dimension, setConfig}: DimensionParams) => {
  onDeleteDimension({config, dimension, setConfig});
  const sortSolver = solverGetter('delete', 'dimension', 'sort');
  return sortSolver({config, dimension, setConfig});
};

const onAddColorDimension = async ({config, dimension, setConfig, reqContext}: DimensionParams) => {
  const colorItemsSolver = solverGetter('add', 'dimension', 'colorItems');
  await colorItemsSolver({dimension, config, setConfig, reqContext});
  setConfig({type: CONFIG.ADD_DIMENSION, payload: {dimension}});
};

const onDeleteColorDimension = ({config, dimension, setConfig}: DimensionParams) => {
  const colorItemsSolver = solverGetter('delete', 'dimension', 'colorItems');
  const sortSolver = solverGetter('delete', 'dimension', 'sort');

  onDeleteDimension({
    config,
    dimension,
    setConfig,
  });
  colorItemsSolver({
    config,
    dimension,
    setConfig,
  });
  sortSolver({
    config,
    dimension,
    setConfig,
  });
};

const onAddMeasure = ({measure, config, setConfig}: MeasureParams) => {
  const colorItemsSolver = solverGetter('add', 'measure', 'colorItems');
  colorItemsSolver({
    config,
    measure,
    setConfig,
  });
  setConfig({type: CONFIG.ADD_MEASURE, payload: measure});
};

const onDeleteMeasure = ({measure, config, setConfig}: MeasureParams) => {
  const sortSolver = solverGetter('delete', 'measure', 'sort');
  const colorItemsSolver = solverGetter('delete', 'measure', 'colorItems');
  colorItemsSolver({config, measure, setConfig});
  sortSolver({config, measure, setConfig});
};

const configHandler = (config: WidgetConfig) => {
  let copiedConfig = sortHandler(config);
  copiedConfig.filter = orFilterGetter(config.filter);
  return copiedConfig;
};

const settings = makeSetting({
  type: 'StackedBarChart',
  icon: `<svg  focusable="false" viewBox="0 0 48 48" aria-hidden="true" role="presentation"><g id="icon-chart-bar"><rect x="0" y="46" width="48" height="2"></rect><rect x="2" y="23" width="8" height="5"></rect><rect x="2" y="38" width="8" height="4"></rect><rect x="2" y="19" width="8" height="3"></rect><rect x="2" y="29" width="8" height="8"></rect><rect x="26" y="1" width="8" height="5"></rect><rect x="26" y="36" width="8" height="6"></rect><rect x="26" y="7" width="8" height="6"></rect><rect x="26" y="14" width="8" height="21"></rect><rect x="14" y="12" width="8" height="2"></rect><rect x="14" y="23" width="8" height="14"></rect><rect x="14" y="38" width="8" height="4"></rect><rect x="14" y="15" width="8" height="7"></rect><rect x="38" y="13" width="8" height="15"></rect><rect x="38" y="37" width="8" height="5"></rect><rect x="38" y="7" width="8" height="5"></rect><rect x="38" y="29" width="8" height="7"></rect></g></svg>`,
  dimensions: [
    {
      type: RequiredType.REQUIRED,
      key: 'x',
      short: 'xaxis',
      columnTypes: [COLUMN_TYPE.TEXT],
      onAdd: onAddDimension,
      onDelete: onDeleteXDimension,
    },
    {
      type: RequiredType.OPTION,
      key: 'color',
      short: 'color',
      columnTypes: [COLUMN_TYPE.TEXT],
      onAdd: onAddColorDimension,
      onDelete: onDeleteColorDimension,
    },
  ],
  measures: [
    {
      type: RequiredType.REQUIRED,
      key: 'y',
      short: 'yaxis',
      columnTypes: [COLUMN_TYPE.NUMBER, COLUMN_TYPE.TEXT],
      onAdd: onAddMeasure,
      onDelete: onDeleteMeasure,
    },
  ],
  enable: true,
  configHandler: configHandler,
});

export default settings;
