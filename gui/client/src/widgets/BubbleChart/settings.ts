import {makeSetting} from '../../utils/Setting';
import {onAddDimension, onDeleteDimension} from '../Utils/settingHelper';
import {cloneObj} from '../../utils/Helpers';
import {orFilterGetter} from '../../utils/Filters';
import {RequiredType, COLUMN_TYPE} from '../../utils/Consts';

const bubbleConfigHandler = (config: any) => {
  let newConfig = cloneObj(config);
  newConfig.filter = orFilterGetter(config.filter);
  return newConfig;
};

const settings = makeSetting({
  type: 'BubbleChart',
  icon: `<svg focusable="false" viewBox="0 0 48 48" aria-hidden="true" role="presentation"><g id="icon-chart-scatter"><path d="M39,6.5c1.4,0,2.5,1.1,2.5,2.5s-1.1,2.5-2.5,2.5s-2.5-1.1-2.5-2.5S37.6,6.5,39,6.5 M39,4c-2.8,0-5,2.2-5,5s2.2,5,5,5 s5-2.2,5-5S41.8,4,39,4L39,4z"></path><circle cx="8" cy="40" r="4"></circle><path d="M39,34.5c2.5,0,4.5,2,4.5,4.5s-2,4.5-4.5,4.5s-4.5-2-4.5-4.5S36.5,34.5,39,34.5 M39,32c-3.9,0-7,3.1-7,7c0,3.9,3.1,7,7,7s7-3.1,7-7C46,35.1,42.9,32,39,32L39,32z"></path><path d="M21.7,14c0.2-0.7,0.3-1.3,0.3-2c0-4.4-3.6-8-8-8s-8,3.6-8,8c0,3.3,2.1,6.2,5,7.4l0,0c-1.2,1.9-2,4.2-2,6.6 c0,6.6,5.4,12,12,12s12-5.4,12-12C33,19.6,28,14.4,21.7,14z M8.5,12c0-3,2.5-5.5,5.5-5.5S19.5,9,19.5,12c0,0.8-0.2,1.5-0.4,2.2h0 c-2.4,0.4-4.6,1.5-6.3,3.2l0,0C10.3,16.8,8.5,14.6,8.5,12z M21,34c-4.4,0-8-3.6-8-8c0-4.4,3.6-8,8-8s8,3.6,8,8 C29,30.4,25.4,34,21,34z"></path></g></svg>`,
  dimensions: [
    {
      type: RequiredType.REQUIRED_ONE_AT_LEAST,
      columnTypes: [COLUMN_TYPE.NUMBER, COLUMN_TYPE.DATE, COLUMN_TYPE.TEXT],
      onAdd: onAddDimension,
      onDelete: onDeleteDimension,
    },
  ],
  measures: [
    {
      type: RequiredType.REQUIRED,
      key: 'x',
      short: 'xaxis',
      columnTypes: [COLUMN_TYPE.TEXT, COLUMN_TYPE.NUMBER],
    },
    {
      type: RequiredType.REQUIRED,
      key: 'y',
      short: 'yaxis',
      columnTypes: [COLUMN_TYPE.TEXT, COLUMN_TYPE.NUMBER],
    },
    {
      type: RequiredType.OPTION,
      key: 'color',
      short: 'color',
      columnTypes: [COLUMN_TYPE.TEXT, COLUMN_TYPE.NUMBER],
    },
    {
      type: RequiredType.OPTION,
      key: 'size',
      short: 'size',
      columnTypes: [COLUMN_TYPE.TEXT, COLUMN_TYPE.NUMBER],
    },
  ],
  enable: true,
  configHandler: bubbleConfigHandler,
});

export default settings;
