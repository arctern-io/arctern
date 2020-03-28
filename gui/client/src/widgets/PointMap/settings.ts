import {makeSetting} from '../../utils/Setting';
import {orFilterGetter} from '../../utils/Filters';
import {CONFIG, COLUMN_TYPE, RequiredType} from '../../utils/Consts';
import {cloneObj} from '../../utils/Helpers';
import {
  getColType,
  // isTextCol
} from '../../utils/ColTypes';
import {queryDistinctValues, MeasureParams, parseTocolorItems} from '../Utils/settingHelper';
import {DEFAULT_MAX_POINTS_NUM, KEY} from '../Utils/Map';
import {measureGetter} from '../../utils/WidgetHelpers';
import {cleanLastSelfFilter, addSelfFilter} from '../../widgets/Utils/settingHelper';
import {MapMeasure} from '../common/MapChart.type';

// PointMap
const _onAddTextColor = async ({measure, config, setConfig, reqContext}: MeasureParams) => {
  const res = await queryDistinctValues({
    dimension: measure,
    config,
    reqContext,
  });
  const colorItems = parseTocolorItems(res);
  setConfig({payload: {colorItems}});
  addSelfFilter({dimension: measure, setConfig, res});
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

const onDeletePointMapColor = ({setConfig}: any) => {
  setConfig({type: CONFIG.DEL_ATTR, payload: ['colorItems']});
};

const pointMapConfigHandler = (config: any) => {
  let newConfig = cloneObj(config);
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
  let lon = measureGetter(newConfig, KEY.LONGTITUDE) as MapMeasure;
  let lat = measureGetter(newConfig, KEY.LATITUDE) as MapMeasure;
  if (!lon || !lat) {
    return newConfig;
  }

  const {_sw, _ne} = newConfig.bounds;

  let colorMeasure = measureGetter(newConfig, 'color');
  const pointMeasure = {
    expression: 'project',
    value: `ST_Point (${lon.value}, ${lat.value})`,
    as: 'point',
  };
  newConfig.measures = [pointMeasure];
  if (colorMeasure) {
    newConfig.measures.push(colorMeasure);
  }
  newConfig.limit = newConfig.points || DEFAULT_MAX_POINTS_NUM;

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
  return newConfig;
};

const settings = makeSetting({
  type: 'PointMap',
  dimensions: [],
  measures: [
    {
      type: RequiredType.REQUIRED,
      key: KEY.LONGTITUDE,
      short: 'longtitude',
      expressions: ['gis_point_lon'],
      columnTypes: [COLUMN_TYPE.NUMBER],
    },
    {
      type: RequiredType.REQUIRED,
      key: KEY.LATITUDE,
      short: 'latitude',
      expressions: ['gis_point_lat'],
      columnTypes: [COLUMN_TYPE.NUMBER],
    },
    {
      type: RequiredType.OPTION,
      key: 'color',
      short: 'color',
      onAdd: onAddColor,
      expressions: ['project'],
      onDelete: onDeletePointMapColor,
      columnTypes: [COLUMN_TYPE.TEXT, COLUMN_TYPE.NUMBER],
    },
  ],
  icon: `<svg focusable="false" viewBox="0 0 48 48"><g id="icon-chart-pointmap"><path d="M10,18.9c0-1.7-1.3-3-3-3s-3,1.3-3,3s1.3,3,3,3S10,20.5,10,18.9z M7,20.9c-1.1,0-2-0.9-2-2s0.9-2,2-2s2,0.9,2,2S8.1,20.9,7,20.9z"></path><path d="M28.5,21.9c-2,0-3.5,1.5-3.5,3.5s1.5,3.5,3.5,3.5s3.5-1.5,3.5-3.5S30.5,21.9,28.5,21.9z M28.5,27.9c-1.4,0-2.5-1.1-2.5-2.5s1.1-2.5,2.5-2.5S31,24,31,25.4S29.9,27.9,28.5,27.9z"></path><path d="M17,23.6c-1.2,0-2.2,1-2.2,2.2s1,2.2,2.2,2.2s2.2-1,2.2-2.2S18.2,23.6,17,23.6z M17,27.1c-0.7,0-1.2-0.6-1.2-1.2s0.6-1.2,1.2-1.2s1.2,0.6,1.2,1.2S17.7,27.1,17,27.1z"></path><circle cx="10.2" cy="25.1" r="1.2"></circle><circle cx="23.5" cy="31.4" r="1.5"></circle><path d="M13,13.1c-1.2,0-2.2,1-2.2,2.2s1,2.2,2.2,2.2s2.2-1,2.2-2.2S14.2,13.1,13,13.1z M13,16.6c-0.7,0-1.2-0.6-1.2-1.2s0.6-1.2,1.2-1.2s1.2,0.6,1.2,1.2S13.7,16.6,13,16.6z"></path><circle cx="17.5" cy="20.4" r="1.5"></circle><path d="M47.2,11.9l-0.4-0.3l-0.3-1.1l-1.2-0.9l-1.1,0l-1.3,1l-0.4,1.9l-0.1,0.3L42.4,13l-2.1,0.5L39.6,14l-0.8,1.1l-0.2,0.8L37.7,16l-0.9,1.5l-0.3,0.4l-0.5-1.4l-0.3,0l-0.2-1.1l-0.8-0.9l0.1-0.2l-0.2-0.5l-0.8-0.7l-0.9-0.3l-0.8,0l-0.6,0.2l-0.4-0.4l-1.5,0l-0.1-0.9l-0.8-0.3l-0.8,0l-0.8-0.3l-1.1-0.7l-0.9,0.2l-7-0.3l-6.6-1.1L5.7,7.9L5,8.6L4.5,8.3L2.6,9.4L2.7,11l-0.2,0.9L1.1,15L1,15.6v1.1l-0.6,1l-0.2,0.7l0.1,1.8l0.2,0.6l0.3,0.4v0l0,0.2l0.1,1l0.3,1.7l0.1,0.3L2,25.7L2,25.9l0.7,1l0.2,0.2l0.6,0.6l0.3,0.5l0.3,0.2L4.2,29l1.1,1.1l1.5,0.2l0.1,0.2l2.6,1.7l0.4,0.2L13,33l0.7-0.5l0.5,0.1l0.4,0.6l0.4,0.3l0.2,0.1l0.1,0.7l0.7,1l1.3,0.7l1.6-0.4l0-0.1l0,0l0.3,0.8l0.2,0.3l0.6,0.8l0.4,1.1l0.8,0.8l1.7,0.5l1.6-1.2v-1.5l1.6-1.1l0.8-0.5l1.3,0.1l0,0l0.9,0.4l0.9,0l0.9-0.5l0.6-0.7l0.2-0.4l0.1,0l0.7,0.2l1-0.1l0.3,0.3l1.6,0.1l0.2-0.1l0.2,0.2l0,0.5l0.4,0.5l0,0.7l0.6,1l0.1,0.1l0.6,0.7l0.6,0.4l0.3,0.6l2.2,0.2l0.7-1l0.2-0.7v-1.2l-0.1-0.6l-0.8-1.6l-0.2-0.7l-1-1.4l-0.1-0.9l0.4-0.7l0.8-0.9l0.3-0.4l0.1-0.4l0.2-0.1l0.4-0.3l1.1-1.2l0-0.5l0.1-0.1l0.5-0.8l0.2-0.7l-0.1-0.8l-0.4-0.8l0.2-0.1l0.2-1.5l-0.1-0.5l0.3-0.2l0.3-1.2l0-0.5l0.1-0.2l1.3-0.7l0.3-0.2l0.7-0.6l-0.6-1.9l0.1-0.4l0.4-0.3l0.3-0.3l0.1-0.2l0.2-0.1l0.8-0.8l0.3-0.9L47.2,11.9z M46,13.7l-0.4,0.2l-0.1,0.1l-0.3,0.5L44.4,15l-0.1,0.2l-0.2,1.7l0.1,0.2l0.4,0.3l0.1,0.1l0.1,0l-0.3,0.2l-1.6,0.8l-0.1,0.1l-0.6,0.9l0,0.2l0.1,0.3L42.1,21l-0.6-0.1l-0.2,0.4l0.5,0.8l-0.1,0.8l-0.2-0.3l-0.4-1.1l-0.5,0.1l0.2,1.4l1.1,2.2l-0.1,0.6l-0.5,0.4l0,0.4l0.1,0.2l-0.9,1L39.9,28l-0.1,0.1L39.5,29l-0.9,1l-0.7,1.2l0,0.1l0.1,1.6l0,0.1l1.1,1.6l0,0.2l0,0.2l0.9,1.8v1.1l-0.4,0.6l-0.3-0.7l-0.2-0.1l-0.4-0.1l-0.5-0.7l-0.5-0.9l0.2-0.3l-0.1-0.4l-0.3-0.1l-0.1-1.1l-0.2-0.2l-0.4-0.1l-0.5-0.6L36.1,33l-0.5,0l-0.1,0l-0.7,0.5L34.2,33L34,33l-1.5,0.2L32.2,33l-0.2,0l-1.4,0.4l-0.2,0.2l-0.1,0.6l-0.5,0.6l-0.7,0l-0.5-0.6l-0.4,0l-0.1,0.2l-1.6-0.1l-0.2,0l-3.3,2.2L23,36.6v1.8L21.8,38l-0.4-1.2l-0.8-1l-0.5-1.3L20,34.4l-1.5-0.6l-0.3,0.1l-0.5,0.7l-1-0.5l-0.1-1.2l-0.1-0.2l-0.6-0.4l-0.7-1L15,31.3L13.5,31l-0.3,0.2l-0.1,0.3l-2.9-0.7l-2.5-1.6l0-0.2l-0.2-0.2l-1.9-0.3l-0.1-0.9l-0.1-0.2l-0.6-0.5l0-0.2l-0.1-0.2L4,26.1l-0.7-1l-0.6-1.3l-0.3-1.6l-0.1-1v-0.4l0-0.1L1.8,20l-0.1-1.6l0.7-1.2l0-0.1v-1.4L4,12.3l0.2-1.2l0-0.1L4.1,9.8l1.3,0.7l0.4-0.2l0-0.9l5.5,1.4L18,12l7.1,0.3l0.2-0.1l0.1-0.2l1,0.7l1.4,0.6l0.2,0l0.4-0.1l0.2,0.1l-0.2,0.1l-0.8,1.1l0.3,0.4l0.5-0.1l0.2,0.3l0.3,0.1l1.1-0.6l0.4-0.4l0.4,0.3l0.2,0.4l0.3,0.1l1.4-0.5l0.7,0.2l0,0l-0.7,0.1l-0.9,0.4h-0.4l-0.2,0.2l-0.5,1.4l0.3,0.3l0.1,0l-0.3,1.8l0.5,1.6l0.4,0.1l0.7-0.6l0.1-0.2l0.1-1.2l0-0.2l-0.3-0.5l0.2-1.3l0.2-0.1l0.2-0.1l0.3-0.8l0.8,0.2l0.2,0.9l-0.3,0.4l0,0.3l0.2,0.4l0.4,0l0.2-0.3l0.4,0.9l-0.5,1.2l0.1,0.3l0.7,0.3l0.2,0l1.7-1l1-1.1l0.1-0.3l-0.2-0.4l1.2-0.2l0.7-0.4l0.1-0.3l-0.1-0.7l0.7-0.9l2.4-0.6l0.1-0.1l0.6-0.6l0.1-0.1l0.4-2.4l0.7,0l0.4,1.4l0.1,0.1l0.6,0.4L46,13.7z"></path></g></svg>`,
  enable: true,
  isServerRender: true,
  configHandler: pointMapConfigHandler,
});

export default settings;
