import {makeSetting} from '../../utils/Setting';
import {RequiredType, COLUMN_TYPE} from '../../utils/Consts';

const settings = makeSetting({
  type: 'Scatter3d',
  icon: `<svg focusable="false" viewBox="0 0 48 48" aria-hidden="true" role="presentation"><g><circle cx="6" cy="42" r="2"></circle><circle cx="10" cy="38" r="2"></circle><circle cx="18" cy="40" r="2"></circle><circle cx="20" cy="35" r="2"></circle><circle cx="14" cy="32" r="2"></circle><circle cx="26" cy="32" r="2"></circle><circle cx="23" cy="26" r="2"></circle><circle cx="30" cy="22" r="2"></circle><circle cx="32" cy="28" r="2"></circle><circle cx="38" cy="30" r="2"></circle><circle cx="32" cy="34" r="2"></circle><circle cx="30" cy="14" r="2"></circle><circle cx="36" cy="10" r="2"></circle><circle cx="40" cy="20" r="2"></circle></g></svg>`,
  dimensions: [],
  measures: [
    {
      type: RequiredType.REQUIRED,
      key: 'x',
      short: 'x',
      columnTypes: [COLUMN_TYPE.NUMBER],
      expressions: [],
    },
    {
      type: RequiredType.REQUIRED,
      key: 'y',
      short: 'y',
      columnTypes: [COLUMN_TYPE.NUMBER],
      expressions: [],
    },
    {
      type: RequiredType.REQUIRED,
      key: 'z',
      short: 'z',
      columnTypes: [COLUMN_TYPE.NUMBER],
      expressions: [],
    },
    {
      type: RequiredType.OPTION,
      key: 'size',
      short: 'size',
      columnTypes: [COLUMN_TYPE.NUMBER],
      expressions: [],
    },
    {
      type: RequiredType.OPTION,
      key: 'color',
      short: 'color',
      columnTypes: [COLUMN_TYPE.NUMBER],
      expressions: [],
    },
  ],
  enable: true,
  isServerRender: true,
});

export default settings;
