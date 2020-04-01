import {makeSetting} from '../../utils/Setting';
import {queryDistinctValues, DimensionParams} from '../Utils/settingHelper';
import {RequiredType, CONFIG, COLUMN_TYPE} from '../../utils/Consts';
import {getColType} from '../../utils/ColTypes';

const _onAddTextType = async ({dimension, config, setConfig, reqContext}: DimensionParams) => {
  const res = await queryDistinctValues({dimension, config, reqContext});
  dimension.options = res.map((r: any) => r[dimension.value]);
  setConfig({type: CONFIG.ADD_DIMENSION, payload: {dimension}});
};

const onAdd = async ({dimension, config, setConfig, reqContext}: any) => {
  const alreadyExist = !!config.dimensions.find((d: any) => d.value === dimension.value);
  if (alreadyExist) {
    return false;
  }
  if (config.filter && config.filter.range) {
    setConfig({type: CONFIG.DEL_FILTER, payload: {id: config.id, filterKeys: ['range']}});
  }
  switch (getColType(dimension.type)) {
    case 'date':
    case 'number':
      setConfig({type: CONFIG.ADD_DIMENSION, payload: {dimension}});
      break;
    case 'text':
      _onAddTextType({dimension, config, setConfig, reqContext});
      break;
    default:
      break;
  }
};

const settings = makeSetting({
  type: 'SlidePlayer',
  icon: `<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24"><path d="M10 18h4v-2h-4v2zM3 6v2h18V6H3zm3 7h12v-2H6v2z"/><path d="M0 0h24v24H0z" fill="none"/></svg>`,
  enable: true,
  dimensions: [
    {
      type: RequiredType.REQUIRED,
      key: 'x',
      short: 'slide',
      columnTypes: [COLUMN_TYPE.DATE, COLUMN_TYPE.NUMBER, COLUMN_TYPE.TEXT],
      onAdd,
    },
  ],
  measures: [],
});

export default settings;
