import {makeSetting} from '../../utils/Setting';
import {RequiredType, COLUMN_TYPE} from '../../utils/Consts';

const settings = makeSetting({
  type: 'NumberChart',
  dimensions: [],
  measures: [
    {
      type: RequiredType.REQUIRED,
      key: 'value',
      short: 'value',
      columnTypes: [COLUMN_TYPE.NUMBER, COLUMN_TYPE.TEXT],
    },
  ],
  icon: `<svg focusable="false" viewBox="0 0 48 48" aria-hidden="true" role="presentation"><g id="icon-chart-number"><path d="M25.4,31h-4.9l-1.7,9h-3.7l1.7-9h-5.1v-3.5h5.7l1.3-6.9h-5.3v-3.5h6L21.1,8h3.7l-1.7,9.1H28L29.7,8h3.7l-1.7,9.1h4.6v3.5 H31l-1.3,6.9h4.9V31h-5.5l-1.7,9h-3.7L25.4,31z M21.2,27.5h4.9l1.3-6.9h-4.9L21.2,27.5z"></path></g></svg>`,
  enable: true,
});

export default settings;
