import {
  format as d3Format,
  timeFormat,
  timeYear,
  timeMonth,
  timeSecond,
  timeHour,
  timeDay,
  utcWeek,
  timeMinute,
} from 'd3';
import {getColType, getNumType, isFloatCol} from '../utils/ColTypes';
import {Dimension, Measure} from '../types';
import {COLUMN_TYPE} from '../utils/Consts';
import {DAYS, DAYS_CN, MONTHS, QUARTERS, QUARTERS_CN, monthDayGetter} from './Time';

// slightly modified version of d3's default time-formatting to always use abbrev month names
const formatMillisecond = timeFormat('.%L');
const formatSecond = timeFormat('%S');
const formatMinute = timeFormat('%I:%M');
const formatHour = timeFormat('%I %p');
const formatDay = timeFormat('%a %d');
const formatWeek = timeFormat('%b %d');
const formatMonth = timeFormat('%b');
const formatYear = timeFormat('%Y');

const DEFAULT_FORMATTER: Function = (d: string | number | Date | boolean) => d;

export const DATE_UNIT_LEVEL: any = {
  millisecond: 0,
  second: 1,
  minute: 2,
  hour: 3,
  day: 4,
  week: 5,
  month: 6,
  quarter: 7,
  year: 8,
  decade: 9,
  century: 10,
  auto: Infinity,
};

export const DATE_FORMAT = [
  {label: 'auto', value: 'auto', tip: '', unit: 'auto'},
  {label: '%y', value: '%y', tip: '19', unit: 'year'},
  {label: '%m/%d/%y', value: '%b/%d/%y', tip: '04/01/19', unit: 'day'},
  {label: '%y-%m-%d', value: '%y-%m-%d', tip: '19-01-01', unit: 'day'},
  {label: '%b', value: '%b', tip: 'Apr', unit: 'month'},
  {label: '%b %d', value: '%b %d', tip: 'Apr 12', unit: 'day'},
  {label: '%a %d', value: '%a %d', tip: 'Mon 27', unit: 'day'},
  {label: '%I%p, %d', value: '%I%p, %d', tip: '11AM, 12', unit: 'hour'},
  {label: '%I%p', value: '%I%p', tip: '11PM', unit: 'hour'},
  {label: '%X', value: '%X', tip: '03AM:46:25', unit: 'second'},
  {label: '%H:%M:%S', value: '%H:%M:%S', tip: '15:46:25', unit: 'second'},
  // { label: "%Y", value: "%Y", tip: "2019" },
  // { label: "%B %d, %Y", value: "%B %d, %Y", tip: "April 01, 2019" },
  // { label: "%Y-%m-%d", value: "%Y-%m-%d", tip: "2019-01-01" },
  // { label: "%m/%d/%Y", value: "%m/%d/%Y", tip: "09/24/2019" },
  // { label: "%B", value: "%B", tip: "April" },
  // { label: "%A", value: "%A", tip: "Monday" },
];

export const NUM_FORMAT = [
  {label: 'auto', value: 'auto', tip: ''},
  {label: 'Float', value: ',.2f', tip: '1,234.57'},
  {label: 'Integer', value: ',.0f', tip: '1,235'},
  {label: 'SI', value: '.2s', tip: '1.2k'},
  {label: 'Percent', value: ',.2%', tip: '97.23%'},
  {label: 'Currency', value: '-$.2f', tip: '-$1234,57'},
];

export const dateFormat = (d: string | Date | number, f: string = '%Y-%m-%dT%H:%M:%S') => {
  return timeFormat(f)(new Date(d));
};

export const autoNumDimensionFormat = (dimension: Dimension) => {
  const {extent = [1, 100]} = dimension;
  let formatter = DEFAULT_FORMATTER;
  const max = extent[1] as number;
  const min = extent[0] as number;
  if (Math.abs(max) < 1000) {
    if (max - min <= 0.02) {
      formatter = d3Format('.4f');
    } else if (max - min <= 0.2) {
      formatter = d3Format('.3f');
    } else if (max - min <= 1.1) {
      formatter = d3Format('.2f');
    } else if (max - min < 100) {
      formatter = d3Format('.1f');
    } else if (max - min < 1000) {
      formatter = d3Format('.0f');
    }
  } else {
    formatter = (d: number) => {
      const abs = Math.abs(d);
      if (abs < 1000) {
        return d3Format(',.2f')(d);
      } else {
        return d3Format(',.2s')(d);
      }
    };
  }
  return formatter;
};

const _autoMeasureFormat = (measure: Measure) => {
  const {expression = '', type} = measure;
  return (num: number) => {
    if (num > 1000 * 1000) {
      return d3Format(',.2s')(num);
    }
    switch (expression) {
      case 'avg':
        return d3Format(',.2f')(num);
      case 'min':
      case 'max':
      case 'sum':
        const numType = getNumType(type);
        return d3Format(numType === 'float' ? ',.2f' : ',.0f')(num);
      case 'unique':
      case 'stddev':
      default:
        return d3Format(isFloatCol(type) ? ',.2f' : ',.0f')(num);
    }
  };
};
const _autoNumFormat = (item: Dimension | Measure): Function => {
  return (item as Dimension).extent
    ? autoNumDimensionFormat(item as Dimension)
    : _autoMeasureFormat(item as Measure);
};

const _getLanguage = () => navigator.language;
const _localDateTimeFormat = (d: Date | string) => timeFormat('%H:%M %p %m/%d/%Y')(new Date(d));
const _binnedValueFormate = (timeBin: string) => {
  switch (timeBin) {
    case 'decade':
    case 'year':
      return (date: string | Date) => timeFormat('%Y')(new Date(date));
    case 'quarter':
      return (date: string | Date) => {
        const language = _getLanguage();
        const milli = new Date(date).getTime() - 10000;
        const vanilli = new Date(milli);
        // calculate the month (0-11) based on the new date
        const mon = vanilli.getMonth();
        const yr = vanilli.getFullYear();
        // return appropriate quarter for that month
        if (mon <= 2) {
          return language === 'en' ? `Q1/${yr}` : `${yr}年第一季度`;
        } else if (mon <= 5) {
          return language === 'en' ? `Q2/${yr}` : `${yr}年第二季度`;
        } else if (mon <= 8) {
          return language === 'en' ? `Q3/${yr}` : `${yr}年第三季度`;
        } else {
          return language === 'en' ? `Q4/${yr}` : `${yr}第四季度`;
        }
      };
    case 'month':
      return (date: string | Date) => timeFormat('%x')(new Date(date));
    case 'week':
      return (date: string | Date) => timeFormat('%x')(new Date(date));
    case 'day':
      return (date: string | Date) => timeFormat('%x %H:%M')(new Date(date));
    case 'hour':
      return (date: string | Date) => timeFormat('%H:%M')(new Date(date));
    case 'minute':
      return (date: string | Date) => timeFormat('%M:%S')(new Date(date));
    default:
      return (date: string | Date) => _localDateTimeFormat(date);
  }
};
const _binnedAxisFormate = (timeBin: string) => {
  switch (timeBin) {
    case 'decade':
    case 'year':
      return (date: string | Date) => timeFormat('%Y')(new Date(date));
    case 'quarter':
      return (date: string | Date) => {
        const language = _getLanguage();
        const milli = new Date(date).getTime() - 10000;
        const vanilli = new Date(milli);
        // calculate the month (0-11) based on the new date
        const mon = vanilli.getMonth();
        const yr = vanilli.getFullYear();
        // return appropriate quarter for that month
        if (mon <= 2) {
          return language === 'en' ? `Q1/${yr}` : `${yr}第一季度`;
        } else if (mon <= 5) {
          return language === 'en' ? `Q2/${yr}` : `${yr}第二季度`;
        } else if (mon <= 8) {
          return language === 'en' ? `Q3/${yr}` : `${yr}第三季度`;
        } else {
          return language === 'en' ? `Q4/${yr}` : `${yr}第四季度`;
        }
      };
    case 'month':
      return (date: string | Date) => timeFormat('%Y-%b')(new Date(date));
    case 'week':
      return (date: string | Date) => timeFormat('%x')(new Date(date));
    case 'day':
      return (date: string | Date) => timeFormat('%x')(new Date(date));
    case 'hour':
      return (date: string | Date) => timeFormat('%H')(new Date(date));
    case 'minute':
      return (date: string | Date) => timeFormat('%M')(new Date(date));
    default:
      return (date: string | Date) => _localDateTimeFormat(date);
  }
};

const _autoBinnedFormate = (timeBin: string, valType: string) => {
  switch (valType) {
    case 'value':
      return _binnedValueFormate(timeBin);
    case 'axis':
      return _binnedAxisFormate(timeBin);
    default:
      return DEFAULT_FORMATTER;
  }
};

const _autoDateFormat = (dimension: Dimension, valType: string) => {
  return dimension.isBinned ? _autoBinnedFormate(dimension.timeBin!, valType) : defaultDateFormat;
};
const _isAutoFormat = (item: any) => !item.format || item.format === 'auto';

const _autoFormatGetter = (item: Dimension | Measure, valType: string) => {
  const dataType = getColType(item.type);
  switch (dataType) {
    case COLUMN_TYPE.DATE:
      const formatter = _autoDateFormat(item as Dimension, valType);
      return (d: string | Date) => formatter(new Date(d));
    case COLUMN_TYPE.NUMBER:
      return _autoNumFormat(item);
    case COLUMN_TYPE.TEXT:
      return (item as Measure).expression ? _countDistinctFormat : DEFAULT_FORMATTER;

    default:
      return DEFAULT_FORMATTER;
  }
};

const _extractFormat = (dimension: Dimension) => {
  const language = _getLanguage();
  switch (dimension.timeBin) {
    case 'quarter':
      return (date: number) => (language === 'en' ? QUARTERS[date - 1] : QUARTERS_CN[date - 1]);
    case 'month':
      return (date: number) => (language === 'en' ? MONTHS[date] : `${date}月`);
    case 'day':
      return (date: number) => (language === 'en' ? monthDayGetter(date - 0) : `${date}日`);
    case 'isodow':
      return (date: number) => (language === 'en' ? DAYS[date - 1] : DAYS_CN[date - 1]);
    case 'hour':
      return (date: number | string) => `${date}${language === 'en' ? 'H' : '点'}`;
    case 'minute':
      return (date: number | string) => `${date}${language === 'en' ? 'm' : '分'}`;
    case 'second':
      return (date: number | string) => `${date}${language === 'en' ? 's' : '秒'}`;
    default:
      return (d: number) => d - 0;
  }
};

const _countDistinctFormat = (num: number) => d3Format(num > 1000 * 1000 ? '.2s' : ',.0f')(num);
export function formatterGetter(item: Dimension | Measure, valType: string = 'value'): Function {
  if (!item) {
    return DEFAULT_FORMATTER;
  }
  const {type, extract = false, format} = item as Dimension;
  const {expression} = item as Measure;

  if (extract) {
    return _extractFormat(item as Dimension);
  }
  if (_isAutoFormat(item)) {
    return _autoFormatGetter(item, valType);
  }
  const dataType = getColType(type);
  switch (dataType) {
    case COLUMN_TYPE.DATE:
      const formatter = timeFormat(format);
      return (d: string | Date) => formatter(new Date(d));
    case COLUMN_TYPE.NUMBER:
      return d3Format(format);
    case COLUMN_TYPE.TEXT:
      return expression ? _countDistinctFormat : DEFAULT_FORMATTER;
    default:
      return DEFAULT_FORMATTER;
  }
}

function defaultDateFormat(date: Date) {
  /* eslint-disable no-nested-ternary */
  return (timeSecond(date) < date
    ? formatMillisecond
    : timeMinute(date) < date
    ? formatSecond
    : timeHour(date) < date
    ? formatMinute
    : timeDay(date) < date
    ? formatHour
    : timeMonth(date) < date
    ? utcWeek(date) < date
      ? formatDay
      : formatWeek
    : timeYear(date) < date
    ? formatMonth
    : formatYear)(date);
  /* eslint-enable no-nested-ternary */
}

type RangeData = {
  dimension: Dimension;
  [key: string]: any;
};
export const rangeFormatter = (rangeDatas: RangeData[]) => {
  const values = rangeDatas.map((rangeData: RangeData) => {
    const {dimension, data} = rangeData;
    const formatter = formatterGetter(dimension);
    let res: string = '';
    if (data.length === 1) {
      res = formatter(data[0]);
    } else {
      res = `${formatter(data[0])} ~ ${formatter(data[1])}`;
    }
    return res;
  });
  return values.join(' / ');
};
