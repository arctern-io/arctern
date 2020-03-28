import {makeSetting} from '../../utils/Setting';
import {Dimension, ConfigHandler} from '../../types';
import {RequiredType, CONFIG, COLUMN_TYPE} from '../../utils/Consts';
import {getColType} from '../../utils/ColTypes';
import {cloneObj} from '../../utils/Helpers';
import {queryDistinctValues, DimensionParams} from '../Utils/settingHelper';
import {parseExpression} from '../../utils/WidgetHelpers';
const _onAddTextType = async ({dimension, config, setConfig, reqContext}: DimensionParams) => {
  const res = await queryDistinctValues({dimension, config, reqContext});
  dimension.options = res.map((r: any) => r[dimension.value]);
  setConfig({type: CONFIG.ADD_DIMENSION, payload: {dimension}});
};

const onAdd = async ({dimension, config, setConfig, reqContext}: any) => {
  //TODO: move this logic to setting later
  const alreadyExist = !!config.dimensions.find((d: any) => d.value === dimension.value);
  if (alreadyExist) {
    return false;
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

const onDelete = ({dimension, config, setConfig}: any) => {
  const filterKeys = Object.keys(config.filter).filter((key: string) => {
    const regex = new RegExp(dimension.as);
    return regex.test(key);
  });
  setConfig({type: CONFIG.DEL_FILTER, payload: {filterKeys: filterKeys}});
};

const configHandler: ConfigHandler = config => {
  let newConfig = cloneObj(config);
  const {filter} = newConfig;
  const filterKeys = Object.keys(filter);
  let groupedFilter: any = {};
  config.dimensions.forEach((d: Dimension) => {
    const {type, as} = d;
    switch (getColType(type)) {
      case 'text':
        if (filter[as]) {
          groupedFilter[as] = filter[as];
        }
        break;
      case 'number':
      case 'date':
        let orGrp: any[] = [];
        const relatedKeys = filterKeys.filter((key: string) => {
          const regex = new RegExp(as);
          return regex.test(key);
        });
        relatedKeys.forEach((key: string) => {
          orGrp.push(parseExpression(filter[key].expr));
        });
        if (orGrp.length) {
          groupedFilter[as] = {
            type: 'filter',
            expr: `(${orGrp.join(`) OR (`)})`,
          };
        }
        break;
      default:
        break;
    }
  });
  newConfig.filter = groupedFilter;
  return newConfig;
};
const settings = makeSetting({
  type: 'FilterWidget',
  icon: `<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24"><path d="M10 18h4v-2h-4v2zM3 6v2h18V6H3zm3 7h12v-2H6v2z"/><path d="M0 0h24v24H0z" fill="none"/></svg>`,
  enable: true,
  dimensions: [
    {
      type: RequiredType.REQUIRED_ONE_AT_LEAST,
      columnTypes: [COLUMN_TYPE.DATE, COLUMN_TYPE.NUMBER, COLUMN_TYPE.TEXT],
      onAdd,
      onDelete,
    },
  ],
  measures: [],
  configHandler,
});

export default settings;
