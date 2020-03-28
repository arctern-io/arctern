import {makeSetting} from '../../utils/Setting';
import {cloneObj} from '../../utils/Helpers';
import {measureGetter} from '../../utils/WidgetHelpers';
import {vegaPointGen, vegaGen, vegaPointWeihtedGen} from '../Utils/Vega';
import {DEFAULT_MAX_POINTS_NUM} from '../Utils/Map';
import {CONFIG, RequiredType, COLUMN_TYPE} from '../../utils/Consts';
import {
  cleanLastSelfFilter,
  addSelfFilter,
  parseTocolorItems,
} from '../../widgets/Utils/settingHelper';
import {getColType, isTextCol} from '../../utils/ColTypes';
import {queryDistinctValues} from '../Utils/settingHelper';

export const getColorTypes = (colorMeasure: any) => {
  return [!!colorMeasure ? 'gradient' : 'solid'];
};
const onAddScatterChartDomain = ({measure, config, setConfig, reqContext}: any) => {
  const sql = `SELECT MIN(${measure.value}) AS min, MAX(${measure.value}) AS max from ${config.source}`;
  reqContext.generalRequest({id: 'choro-min-max', sql}).then((res: any) => {
    if (res && res[0]) {
      const {min, max} = res[0];
      measure = {...cloneObj(measure), domain: [min, max], staticDomain: [min, max]};

      setConfig({payload: measure, type: CONFIG.ADD_MEASURE});
    }
  });
};
const _onAddNumColor = async ({measure, config, setConfig, reqContext}: any) => {
  const {filter = {}} = config;
  Object.keys(filter).forEach((filterKey: string) => {
    if (!filter[filterKey].expr.geoJson) {
      setConfig({type: CONFIG.DEL_FILTER, payload: [filterKey]});
    }
  });
  reqContext.numMinMaxValRequest(measure.value, config.source).then((res: any) => {
    const ruler = res;
    setConfig({type: CONFIG.ADD_RULER, payload: ruler});
    setConfig({type: CONFIG.ADD_RULERBASE, payload: ruler});
  });
};
const _onAddTextColor = async ({measure, config, setConfig, reqContext}: any) => {
  const res = await queryDistinctValues({
    dimension: measure,
    config,
    reqContext,
  });
  const colorItems = parseTocolorItems(res);
  setConfig({type: CONFIG.ADD_COLORITEMS, payload: colorItems});
  addSelfFilter({dimension: measure, setConfig, res});
};
const onAddColor = async ({measure, config, setConfig, reqContext}: any) => {
  cleanLastSelfFilter({dimension: measure, setConfig, config});
  setConfig({type: CONFIG.DEL_ATTR, payload: ['colorItems']});

  const dataType = getColType(measure.type);
  switch (dataType) {
    case 'text':
      setConfig({type: CONFIG.DEL_ATTR, payload: ['colorKey']});
      await _onAddTextColor({measure, config, setConfig, reqContext});
      break;
    case 'number':
      await _onAddNumColor({measure, config, setConfig, reqContext});
      break;
    default:
      break;
  }
  setConfig({type: CONFIG.ADD_MEASURE, payload: measure});
};

const scatterConfigHandler = (config: any) => {
  const copiedConfig = cloneObj(config);
  const xMeasure = measureGetter(config, 'x');
  const yMeasure = measureGetter(config, 'y');
  const colorMeasure = measureGetter(config, 'color');
  const sizeMeasure = measureGetter(config, 'size');

  if (!xMeasure || !yMeasure) {
    return copiedConfig;
  }

  // Put limit
  copiedConfig.limit = copiedConfig.points || DEFAULT_MAX_POINTS_NUM;
  const as = config.measures.map((m: any) => m.as);
  const isGradientColor = !!colorMeasure;
  const isPointVega = colorMeasure && isTextCol(colorMeasure.type);
  const vega = isPointVega
    ? vegaPointGen(copiedConfig)
    : isGradientColor
    ? vegaPointWeihtedGen(copiedConfig, getColorTypes(colorMeasure))
    : vegaGen(copiedConfig, getColorTypes(colorMeasure));

  const param = `rect.x, rect.y${colorMeasure ? `, rect.${colorMeasure.as}` : ''}${
    sizeMeasure ? `, rect.${sizeMeasure.as}` : ''
  }`;
  copiedConfig.renderSelect = `plot_scatter_2d(${param}, '${vega}')`;
  copiedConfig.renderAs = `rect(${as.join(' , ')})`;
  return copiedConfig;
};

const onAfterSqlCreate = (sql: string, config: any): string =>
  `SELECT ${config.renderSelect} from (${sql}) as ${config.renderAs}`;

const settings = makeSetting({
  type: 'ScatterChart',
  icon: `<svg focusable="false" viewBox="0 0 48 48" aria-hidden="true" role="presentation"><g><circle cx="6" cy="42" r="2"></circle><circle cx="10" cy="38" r="2"></circle><circle cx="18" cy="40" r="2"></circle><circle cx="20" cy="35" r="2"></circle><circle cx="14" cy="32" r="2"></circle><circle cx="26" cy="32" r="2"></circle><circle cx="23" cy="26" r="2"></circle><circle cx="30" cy="22" r="2"></circle><circle cx="32" cy="28" r="2"></circle><circle cx="38" cy="30" r="2"></circle><circle cx="32" cy="34" r="2"></circle><circle cx="30" cy="14" r="2"></circle><circle cx="36" cy="10" r="2"></circle><circle cx="40" cy="20" r="2"></circle></g></svg>`,
  dimensions: [],
  measures: [
    {
      type: RequiredType.REQUIRED,
      key: 'x',
      short: 'xaxis',
      columnTypes: [COLUMN_TYPE.NUMBER],
      onAdd: onAddScatterChartDomain,
      expressions: ['gis_discrete_trans_scale_w'],
    },
    {
      type: RequiredType.REQUIRED,
      key: 'y',
      short: 'yaxis',
      columnTypes: [COLUMN_TYPE.NUMBER],
      onAdd: onAddScatterChartDomain,
      expressions: ['gis_discrete_trans_scale_h'],
    },
    {
      type: RequiredType.OPTION,
      key: 'color',
      short: 'color',
      expressions: ['project'],
      columnTypes: [COLUMN_TYPE.NUMBER, COLUMN_TYPE.TEXT],
      onAdd: onAddColor,
    },
  ],
  enable: true,
  configHandler: scatterConfigHandler,
  onAfterSqlCreate,
});

export default settings;
