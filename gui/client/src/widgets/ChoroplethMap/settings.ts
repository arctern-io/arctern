import {cloneObj} from '../../utils/Helpers';
import {orFilterGetter} from '../../utils/Filters';
import {makeSetting} from '../../utils/Setting';
import {CONFIG, COLUMN_TYPE, RequiredType} from '../../utils/Consts';
import {KEY as MAPKEY} from '../Utils/Map';
import {measureGetter, dimensionGetter, getExpression} from '../../utils/WidgetHelpers';
import {ChoroplethMapConfig} from './types';
import {MapMeasure} from '../common/MapChart.type';
import {MeasureParams} from '../Utils/settingHelper';
const onAddChoroplethMapColor = async ({measure, config, setConfig, reqContext}: MeasureParams) => {
  const buildDimension = dimensionGetter(config, 'wkt');
  if (buildDimension) {
    const {as} = measure;
    let expression = getExpression(measure);
    // cause megawise is not working for text group subQuery at the moment, change to one Query when it's ready;
    const minSql = `SELECT ${expression} FROM ${config.source} GROUP BY ${buildDimension.value} ORDER BY ${as} ASC LIMIT 1`;
    const maxSql = `SELECT ${expression} FROM ${config.source} GROUP BY ${buildDimension.value} ORDER BY ${as} DESC LIMIT 1`;
    const rulerBaseMin = await reqContext.generalRequest(minSql);
    const rulerBaseMax = await reqContext.generalRequest(maxSql);
    console.info(rulerBaseMin, rulerBaseMax);
    const ruler = {min: rulerBaseMin[0][as], max: rulerBaseMax[0][as]};
    const rulerBase = {min: rulerBaseMin[0][as], max: rulerBaseMax[0][as]};
    setConfig({type: CONFIG.ADD_RULER, payload: ruler});
    setConfig({type: CONFIG.ADD_RULERBASE, payload: rulerBase});
    setConfig({type: CONFIG.ADD_MEASURE, payload: measure});
  }
};

const choroplethMapConfigHandler = <ChoroplethMapConfig>(config: ChoroplethMapConfig) => {
  let newConfig = cloneObj(config);
  // Start: handle map bound
  if (!newConfig.bounds) {
    newConfig.bounds = {
      _sw: {
        lng: -73.5,
        lat: 40.1,
      },
      _ne: {
        lng: -70.5,
        lat: 41.1,
      },
    };
  }
  let lon = measureGetter(newConfig, MAPKEY.LONGTITUDE) as MapMeasure;
  let lat = measureGetter(newConfig, MAPKEY.LATITUDE) as MapMeasure;
  let wkt = dimensionGetter(newConfig, 'wkt')!;
  const {value, as, expression} = wkt;
  let wktM = {
    value,
    as,
    expression,
  };
  if (!newConfig.bounds) {
    newConfig.bounds = {
      _sw: {
        lng: -73.5,
        lat: 40.1,
      },
      _ne: {
        lng: -70.5,
        lat: 41.1,
      },
    };
    // return newConfig;
  }
  const {_sw, _ne} = newConfig.bounds;
  let colorM = measureGetter(newConfig, 'w');

  newConfig.selfFilter.bounds = {
    type: 'filter',
    expr: {
      type: 'st_within',
      x: lon.value,
      y: lat.value,
      px: [_sw.lng, _sw.lng, _ne.lng, _ne.lng, _sw.lng],
      py: [_sw.lat, _ne.lat, _ne.lat, _sw.lat, _sw.lat],
    },
  };

  newConfig.filter = orFilterGetter(newConfig.filter);

  // gen vega
  newConfig.measures = [colorM, wktM];
  return newConfig;
};

const settings = makeSetting<ChoroplethMapConfig>({
  type: 'ChoroplethMap',
  dimensions: [
    {
      type: RequiredType.REQUIRED,
      key: 'wkt',
      short: 'building',
      isNotUseBin: true,
      columnTypes: [COLUMN_TYPE.NUMBER, COLUMN_TYPE.TEXT],
    },
  ],
  measures: [
    {
      type: RequiredType.REQUIRED,
      key: MAPKEY.LONGTITUDE,
      short: 'longtitude',
      expressions: ['project'],
      columnTypes: [COLUMN_TYPE.NUMBER],
    },
    {
      type: RequiredType.REQUIRED,
      key: MAPKEY.LATITUDE,
      short: 'latitude',
      expressions: ['project'],
      columnTypes: [COLUMN_TYPE.NUMBER],
    },
    {
      type: RequiredType.REQUIRED,
      key: 'm',
      short: 'color',
      onAdd: onAddChoroplethMapColor,
      expressions: ['project'],
      columnTypes: [COLUMN_TYPE.NUMBER],
    },
  ],
  icon: `<svg focusable="false" viewBox="0 0 48 48"><g><path d="M47.2,11.9l-0.4-0.3l-0.3-1.1l-1.2-0.9h-1.1l-1.3,1l-0.4,1.9l-0.1,0.3V13l-2.1,0.5L39.6,14l-0.8,1.1l-0.2,0.8L37.7,16l-0.9,1.5l-0.3,0.4L36,16.5h-0.3l-0.2-1.1l-0.8-0.9l0.1-0.2l-0.2-0.5l-0.8-0.7l-0.9-0.3h-0.8L31.5,13l-0.4-0.4h-1.5l-0.1-0.9l-0.8-0.3h-0.8l-0.8-0.3L26,10.4l-0.9,0.2l-7-0.3l-6.6-1.1L5.7,7.9L5,8.6L4.5,8.3L2.6,9.4L2.7,11l-0.2,0.9L1.1,15L1,15.6v1.1l-0.6,1l-0.2,0.7l0.1,1.8l0.2,0.6l0.3,0.4v0.2l0.1,1l0.3,1.7l0.1,0.3L2,25.7v0.2l0.7,1l0.2,0.2l0.6,0.6l0.3,0.5l0.3,0.2L4.2,29l1.1,1.1l1.5,0.2l0.1,0.2l2.6,1.7l0.4,0.2L13,33l0.7-0.5l0.5,0.1l0.4,0.6l0.4,0.3l0.2,0.1l0.1,0.7l0.7,1l1.3,0.7l0.5-0.1l0,0l0.5-0.1l0,0l0.6-0.2v-0.1l0.3,0.8l0.2,0.3l0.6,0.8l0.4,1.1l0.8,0.8l1.7,0.5l0.7-0.5l0,0l0,0l0.9-0.7v-1.5l1.6-1.1l0.8-0.5l1.3,0.1l0.9,0.4H30l0.9-0.5l0.6-0.7l0.2-0.4h0.1l0.7,0.2l1-0.1l0.3,0.3l1.6,0.1l0.2-0.1l0.2,0.2v0.5l0.4,0.5v0.7l0.6,1l0.1,0.1l0.6,0.7l0.6,0.4l0.3,0.6l2.2,0.2l0.7-1l0.2-0.7v-1.2l-0.1-0.6l-0.8-1.6l-0.2-0.7l-1-1.4l-0.1-0.9l0.4-0.7l0.8-0.9l0.3-0.4l0.1-0.4l0.2-0.1l0.4-0.3l1.1-1.2v-0.5l0.1-0.1l0.5-0.8l0.2-0.7l-0.1-0.8l-0.4-0.8l0.2-0.1l0.2-1.5l-0.1-0.5l0.3-0.2l0.3-1.2v-0.5l0.1-0.2l1.3-0.7l0.3-0.2l0.7-0.6l-0.6-1.9l0.1-0.4l0.4-0.3l0.3-0.3l0.1-0.2l0.2-0.1l0.8-0.8l0.3-0.9L47.2,11.9z M44.1,16.5l-1.1,0.4l-0.2-0.6l1.3-0.4L44.1,16.5z M42.3,20l-0.1,0.3L42.1,20L42,18.9l0.1-0.4l0.5,0.4l-0.4,0.6v0.2L42.3,20z M37.1,24.9L36.7,25l0.3-0.5l1.3-0.3l0.3-0.2l1.4-3l1.5-0.5l0.2,0.4l-0.2,0l-0.2,0.4l0.5,0.8l-0.1,0.8l-0.2-0.3l-0.4-1.1l-0.5,0.1l0.2,1.4l0.4,0.9L37.1,24.9z M37.9,31.7l-2.6,0.2l-0.1-1.2l-0.9-2.5l1.5-0.4l2.4,2.8l-0.3,0.6v0.1L37.9,31.7z M29.3,28.2L28.7,30l0,0.2l0,0.4l-2.1,0.2v-0.4L26.4,30l-0.1,0l0.1-2l-0.2-0.6l2.7-0.3l0,0.2l0.3,0.3l0.2,0L29.3,28.2L29.3,28.2z M10.3,19.1l1.5,0.4l-0.1,0.7l0.3,0.4l1.3,0.3L13,23.3l-0.2,1.7l-3.4-0.8L10.3,19.1z M8.2,25.3L8,25.6l-0.2,0.3l-3-5.1l0.8-2.9l4,1L8.2,25.3z M12.5,19.9l0.6-3.5l4.7,0.3l-0.2,1.9l0,0l0,0l-0.2,2.1l-2.8-0.3l-0.8-0.1L12.5,19.9z M18.7,15.4l4.7,0.3l0,0.3l0.3,0.4l-0.2,2.4l-1-0.3l-4.2-0.2l0.2-1.9l0-0.2L18.7,15.4z M27.9,18.8l0.1,0.2l0.8,1l-0.6,1.4l-3.2,0l-0.4-2l-0.1-0.2l0-0.6l3.5-0.1L27.9,18.8z M29.5,20.1l0-0.4l0,0l1.5-0.4l0.4,1.2l0,0l0.1,2.8l-0.2,1l-0.6,0.8l-0.9-1.5l-1-1.1l0-0.7L29.5,20.1z M32.5,19.8l0-0.1l0,0.3l0.7,0l0.3,2.8L33,23.9L32,24l0.1-0.6l-0.1-3.1l0.3-0.3L32.5,19.8z M38,18.4l3-0.7l0.4,0.3l-0.2,0.7l0,0.1l0.1,0.9l-1.7,0.5l-1.7,0.4l-0.5-1.6L38,18.4z M37.8,21.5l1.2-0.3l-1.1,2.3l-1,0.2l-0.6-0.9L37.8,21.5z M35.7,23.2l0.7,1l-0.6,1l-4.8,0.8l0,0l0.9-1.1l1.5-0.3l0.3-0.2l0.6-1.1H35.7z M19.7,22.1l0-0.9l-0.3-0.4l-1.1-0.1l0.1-1.6l4,0.1l1.4,0.5l0.5,2.1l0.1,0.3l-3.8-0.1L19.7,22.1z M19.5,25.8l0-0.5l0.2-2.4l5.1,0.1l0.5,0.8V26l-0.8,0L19.5,25.8z M26,26.3v-2.7l-0.1-0.2l-0.3-0.5l-0.4-0.8l2.7,0l0,0.5l0.1,0.3l1.1,1.1l1,1.8l-0.4,1.1l-0.1,0l0-0.3l-0.4-0.3L26,26.5L26,26.3z M14,21l4.9,0.6l-0.1,3.6l0,0.5l-0.5,0l-4.7-0.6L14,21z M25.3,29.5l-0.2,0l-1.9,0.2l-2-0.8l0-1.9l-0.3-0.3l4.4,0.1l0.1,0.3l0,0l0.2,0.9l-0.1,1.7l0,0L25.3,29.5z M29.6,31.8l0-0.2l-0.2-1.4l0.5-1.4l1.5-0.2l0.1,4.6l-0.4,0.1l-1.9-0.3L29.6,31.8z M32.2,28.4l1.5-0.1l0.9,2.5l0.1,1.1l-2.4,0.2L32.2,28.4z M35,27.2L35,27.2l-1.2,0.3l-2.1,0.1l0,0l0,0l-1.5,0.2l0.2-0.5l0,0l0,0l0.2-0.6l5.6-0.9L35,27.2z M39.4,17.2l0.7-0.4l0.1-0.3l-0.1-0.7l0.7-0.9l0.9-0.2l0.8,3.3l-1.3-0.9L41,17l-2.7,0.6l-0.1-0.2L39.4,17.2z M35.6,20.2l1.2-0.7l0.5,1.5l-1.5,1.4h-1.5L34,19.9l0.7,0l0,0l0.7,0.3H35.6z M34,15.8l0.2,0.9l-0.3,0.4v0.3l0.2,0.4h0.4l0.2-0.3l0.4,0.9l-0.3,0.7l-2.3,0.1l0.1-0.6v-0.2l-0.3-0.5l0.2-1.3l0.2-0.1l0.2-0.1l0.3-0.8L34,15.8z M28.9,15.2l1.1-0.6l0.4-0.4l0.4,0.3l0.2,0.4l0.3,0.1l1.4-0.5l0.7,0.2l-0.7,0.1l-0.9,0.4h-0.4l-0.2,0.2l-0.5,1.4l0.3,0.3h0.1l-0.2,1.4l-2,0.5l-0.3-0.4l-0.2-1.3l-0.2-0.3l-1-0.7l0.3-1.1l0.7-0.5l0.1,0l0.2,0.3L28.9,15.2z M25.3,12.2l0.1-0.2l1,0.7l1.4,0.6H28l0.4-0.1l0.2,0.1l-0.2,0.1l-0.6,0.9l-0.7,0.4l-0.2,0.2l-0.4,1.5l0.1,0.4l1.1,0.8l0,0.2l-3.3,0.1l0.1-1.5l-0.1-0.2l-0.2-0.3l0.2-0.2l-0.1-0.3l-0.2-3l1.2,0L25.3,12.2z M23.2,12.2l0.2,2.7l-4.7-0.3l0.2-2.6L23.2,12.2z M10.6,10.6l0.7,0.2L18,12l0.2,0l-0.3,4l-5-0.4l0,0l0,0l-1.2-0.1l-0.4-1.2l0.1-1.2l0-0.2l-0.8-1.7L10.6,10.6z M9.7,11.5l0.9,1.7l-0.1,1.2l0.5,1.6l0.3,0.3l1.1,0.1l-0.4,2.4L8,17.5l0.6-2.1l0.8-1.4l0-0.4l-0.2-0.5l0.6-1.9L9.7,11.5z M5.4,10.5l0.4-0.2V9.4l3.3,0.9l-0.7,2.6l-1-0.2L5,12.3v-0.5l-0.3-0.4l-0.5-0.2l0-0.2V11L4.1,9.8L5.4,10.5z M2.4,15.7L4,12.3L4,12l0.2,0v0.5L4.5,13l3.9,0.7l0.1,0.3l-0.7,1.3l-0.6,2.2L5.4,17l-2.1-0.9L2.4,15.7L2.4,15.7z M6.9,28.7l-1.3-0.2l-0.1-0.9l-0.1-0.2l-0.6-0.5v-0.2l-0.1-0.2L4,26.1l-0.7-1l-0.6-1.3l-0.3-1.6l-0.1-1v-0.4v-0.1L1.8,20l-0.1-1.6l0.7-1.2v-0.1v-0.5l0.4,0.2l2.1,0.9L4,20.7l0,0.3l3.4,5.8l0.2,0.7L6.9,28.7z M10.2,30.8l-2.5-1.6V29l0,0l0.7-1.3l0-0.3l-0.2-0.7l0.7-1l0.2-0.8l2.6,0.6l0.9,0.2l-0.9,5.5L10.2,30.8z M15.2,31.3L15.2,31.3L15.2,31.3l-0.2,0L13.5,31l-0.3,0.2l-0.1,0.3l-0.6-0.1l0.9-5.5l4.4,0.6v0.2l0,0l-0.5,4.8L15.2,31.3z M25.9,34.4l0,0.2L23,36.5v0.1v1.8L21.8,38l-0.4-1.2l-0.8-1l-0.5-1.3L20,34.4l-1.5-0.6l-0.3,0.1l-0.5,0.7l-1-0.5l-0.1-1.2l-0.1-0.2l-0.6-0.4l-0.1-0.2l1.9,0.1l0,0l0,0l0,0l0.4-0.3l0,0l0,0l0,0l0.6-4.9l1.5,0.1l0.3,0L20.3,29l0.2,0.4l0,0l0,0l2.4,1.1l0.2,0l2-0.2l0.6,0.3V32l0.6,1.1L25.9,34.4z M29.8,34.8h-0.7l-0.5-0.6h-0.4l-0.1,0.2l-1.3-0.1l0.4-1.1l0-0.3l-0.6-1v-0.5l2.3-0.2l0.1,0.4L28.3,33l0.3,0.5l1.7,0.3l-0.1,0.4L29.8,34.8z M40,36.8v1.1l-0.4,0.6l-0.3-0.7l-0.2-0.1l-0.4-0.1l-0.5-0.7L37.7,36l0.2-0.3l-0.1-0.4l-0.3-0.1l-0.1-1.1l-0.2-0.2l-0.4-0.1l-0.5-0.6L36.1,33h-0.5h-0.1l-0.7,0.5L34.2,33H34l-1.5,0.2L32.2,33l0-0.1l5.7-0.5l0,0.5V33l1.1,1.6v0.2V35L40,36.8z M39.8,28.1L39.5,29l-0.8,0.9l-1.9-2.2l2.5-0.3l1,0.5L39.9,28L39.8,28.1z M41.8,25.8l-0.5,0.4v0.4l0.1,0.2l-0.5,0.6l-1.4-0.8l-0.2,0L36.2,27l1.2-1.4l4.2-1.1l0.3,0.6L41.8,25.8z M44.5,17.7l-1.1,0.5l-0.2-0.6l1.1-0.4l0.2,0.2l0.1,0.1h0.1L44.5,17.7z M46,13.7l-0.4,0.2L45.5,14l-0.3,0.5L44.4,15l0,0l-1.7,0.5l-0.3-1.1l0.7-0.2l0.1-0.1l0.6-0.6l0.1-0.1l0.4-2.4h0.7l0.4,1.4l0.1,0.1l0.6,0.4L46,13.7z"></path></g></svg>`,
  enable: true,
  configHandler: choroplethMapConfigHandler,
  onAfterSqlCreate: (sql: string, config: ChoroplethMapConfig): string => {
    return `SELECT ${config.renderSelect} from (${sql}) as ${config.renderAs}`;
  },
});

export default settings;
