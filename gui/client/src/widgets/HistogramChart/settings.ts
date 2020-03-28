import {makeSetting} from '../../utils/Setting';
import {cloneObj} from '../../utils/Helpers';
import {dimensionGetter} from '../../utils/WidgetHelpers';
import {solverGetter, genColorItem} from '../Utils/settingHelper';
import {CONFIG, COLUMN_TYPE, RequiredType} from '../../utils/Consts';

const onAddMeasure = ({measure, config, setConfig}: any) => {
  const colorDimension = dimensionGetter(config, 'color');
  if (!colorDimension) {
    setConfig({type: CONFIG.CLERAR_COLORITEMS});
    const colorItem = genColorItem(measure);
    setConfig({type: CONFIG.ADD_COLORITEMS, payload: [colorItem]});
  }
  setConfig({payload: measure, type: CONFIG.ADD_MEASURE});
};

const onDeleteMeasure = ({measure, config, setConfig}: any) => {
  const colorItemsSolver = solverGetter('delete', 'measure', 'colorItems');
  return colorItemsSolver({config, measure, setConfig});
};
// HistogramChart
const onAddColor = async ({config, dimension, setConfig, reqContext}: any) => {
  const colorItemsSolver = solverGetter('add', 'dimension', 'colorItems');
  await colorItemsSolver({dimension, config, setConfig, reqContext});
  setConfig({payload: {dimension}, type: CONFIG.ADD_DIMENSION});
};
const onDeleteColor = ({config, setConfig}: any) => {
  const colorItemsSolver = solverGetter('delete', 'dimension', 'colorItems');
  return colorItemsSolver({config, setConfig});
};

const lineChartHandler = (config: any) => {
  let newConfig = cloneObj(config);
  const xDimension = dimensionGetter(config, 'x')!;
  newConfig.sort = {name: xDimension.as};
  return newConfig;
};

const settings = makeSetting({
  type: 'HistogramChart',
  dimensions: [
    {
      type: RequiredType.REQUIRED,
      key: 'x',
      short: 'xaxis',
      columnTypes: [COLUMN_TYPE.NUMBER, COLUMN_TYPE.DATE],
    },
    {
      type: RequiredType.OPTION,
      key: 'color',
      short: 'color',
      columnTypes: [COLUMN_TYPE.TEXT],
      onAdd: onAddColor,
      onDelete: onDeleteColor,
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
  icon: `<svg focusable="false" viewBox="0 0 24 24" aria-hidden="true" role="presentation"><path d="M5 9.2h3V19H5zM10.6 5h2.8v14h-2.8zm5.6 8H19v6h-2.8z"></path></svg>`,
  enable: true,
  configHandler: lineChartHandler,
});

export default settings;
