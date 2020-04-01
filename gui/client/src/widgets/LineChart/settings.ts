import {makeSetting} from '../../utils/Setting';
import {solverGetter, genColorItem} from '../Utils/settingHelper';
import {cloneObj} from '../../utils/Helpers';
import {dimensionGetter} from '../../utils/WidgetHelpers';
import {CONFIG, RequiredType, COLUMN_TYPE} from '../../utils/Consts';

const onAddColorDimension = async ({config, dimension, setConfig, reqContext}: any) => {
  const colorItemsSolver = solverGetter('add', 'dimension', 'colorItems');
  await colorItemsSolver({dimension, config, setConfig, reqContext});
  setConfig({type: CONFIG.ADD_DIMENSION, payload: {dimension}});
};

const onDeleteColor = ({config, setConfig}: any) => {
  const colorItemsSolver = solverGetter('delete', 'dimension', 'colorItems');
  return colorItemsSolver({config, setConfig});
};

const onAddMeasure = ({config, setConfig, measure}: any) => {
  const colorDimension = dimensionGetter(config, 'color');
  if (!colorDimension) {
    const colorItem = genColorItem(measure);
    setConfig({type: CONFIG.ADD_COLORITEMS, payload: [colorItem]});
  }
  setConfig({type: CONFIG.ADD_MEASURE, payload: measure});
};

const onDeleteMeasure = ({measure, config, setConfig}: any) => {
  const colorItemsSolver = solverGetter('delete', 'measure', 'colorItems');
  return colorItemsSolver({config, measure, setConfig});
};

const lineChartHandler = (config: any) => {
  let newConfig = cloneObj(config);
  const xDimension = dimensionGetter(config, 'x')!;
  newConfig.sort = {name: xDimension.as};
  return newConfig;
};

const settings = makeSetting({
  type: 'LineChart',
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
      onAdd: onAddColorDimension,
      onDelete: onDeleteColor,
    },
  ],
  measures: [
    {
      type: RequiredType.REQUIRED_ONE_AT_LEAST,
      short: 'yaxis',
      columnTypes: [COLUMN_TYPE.NUMBER, COLUMN_TYPE.TEXT],
      onAdd: onAddMeasure,
      onDelete: onDeleteMeasure,
    },
  ],
  icon: `<svg  focusable="false" viewBox="0 0 48 48" aria-hidden="true" role="presentation"><g id="icon-chart-line"><path d="M40.3,22.5l1.7,1.7V42H8.7l11-11l3.5,1.7l3.2,1.6l1.3,0.6l1-1l2.5-2.5L40.3,22.5 M40.3,19.6L29.9,30l-2.5,2.5 l-3.2-1.6l-4.8-2.4L6.9,40.9L4,43.9V44h40V23.4l-3.5-3.5L40.3,19.6L40.3,19.6z"></path><polygon points="44,6 33.7,6 37.4,9.7 25.6,21.6 17.6,17.6 4,31.2 4,36.8 18.4,22.4 26.4,26.4 40.3,12.6 44,16.3 "></polygon></g></svg>`,
  enable: true,
  configHandler: lineChartHandler,
});

export default settings;
