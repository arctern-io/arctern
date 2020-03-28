import {
  timeYear,
  utcYear,
  timeMonth,
  utcMonth,
  timeSecond,
  utcSecond,
  timeMillisecond,
  utcMillisecond,
  timeHour,
  utcHour,
  timeDay,
  utcDay,
  timeWeek,
  utcWeek,
  timeMinute,
  utcMinute,
} from 'd3';

export const DAYS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
export const DAYS_CN = ['周一', '周二', '周三', '周四', '周五', '周六', '周日'];
export const MONTHS = [
  'Jan',
  'Feb',
  'Mar',
  'Apr',
  'May',
  'Jun',
  'Jul',
  'Aug',
  'Sep',
  'Oct',
  'Nov',
  'Dec',
];
export const QUARTERS = ['Q1', 'Q2', 'Q3', 'Q4'];
export const QUARTERS_CN = ['第一季度', '第二季度', '第三季度', '第四季度'];
export const HOURS = [
  '12AM',
  '1AM',
  '2AM',
  '3AM',
  '4AM',
  '5AM',
  '6AM',
  '7AM',
  '8AM',
  '9AM',
  '10AM',
  '11AM',
  '12PM',
  '1PM',
  '2PM',
  '3PM',
  '4PM',
  '5PM',
  '6PM',
  '7PM',
  '8PM',
  '9PM',
  '10PM',
  '11PM',
];
export const monthDayGetter = (num: number) => {
  switch (num) {
    case 1:
    case 11:
    case 21:
    case 31:
      return `${num}st`;
    case 2:
    case 22:
      return `${num}nd`;
    case 3:
    case 23:
      return `${num}rd`;
    default:
      return `${num}th`;
  }
};

export const d3TimeTranslation: {[key: string]: Function} = {
  '1c': timeYear.round,
  '10y': timeYear.round,
  '1y': timeYear.round,
  '1q': timeMonth.round,
  '1mo': timeMonth.round,
  '1s': timeSecond.round,
  '1ms': timeMillisecond.round,
  '1m': timeMinute.round,
  '1h': timeHour.round,
  '1d': timeDay.round,
  '1w': timeWeek.round,
};

export const d3TimeTranslationUTC: {[key: string]: Function} = {
  '1c': utcYear.round,
  '10y': utcYear.round,
  '1y': utcYear.round,
  '1q': utcMonth.round,
  '1mo': utcMonth.round,
  '1s': utcSecond.round,
  '1ms': utcMillisecond.round,
  '1m': utcMinute.round,
  '1h': utcHour.round,
  '1d': utcDay.round,
  '1w': utcWeek.round,
};

export const timeBinMap: {[key: string]: string} = {
  auto: 'auto',
  century: '1c',
  decade: '10y',
  year: '1y',
  quarter: '1q',
  month: '1mo',
  week: '1w',
  day: '1d',
  hour: '1h',
  minute: '1m',
  second: '1s',
  millisecond: '1ms',
};

export const TIME_GAPS: {[key: string]: number} = {
  '1c': 31536000 * 1000,
  '10y': 31536000 * 1000,
  '1y': 31536000 * 1000,
  '1q': 10368000 * 1000,
  '1mo': 2592000 * 1000,
  '1s': 1 * 1000,
  '1ms': 1,
  '1h': 3600 * 1000,
  '1d': 86400 * 1000,
  '1m': 60 * 1000,
  '1w': 604800 * 1000,
};

export enum TimeBin {
  CENTURY = 'century',
  DECADE = 'decade',
  YEAR = 'year',
  QUARTER = 'quarter',
  MONTH = 'month',
  WEEK = 'week',
  DAY = 'day',
  HOUR = 'hour',
  MINUTE = 'minute',
  SECOND = 'second',
  MILLISECONDS = 'milliseconds',
}
export enum ExtractBin {
  CENTURY = 'century',
  DECADE = 'decade',
  YEAR = 'year',
  QUARTER = 'quarter',
  MONTH = 'month',
  DAY = 'day',
  ISODOW = 'isodow',
  HOUR = 'hour',
  MINUTE = 'minute',
  SECOND = 'second',
  MILLISECONDS = 'milliseconds',
}

export const TIME_BIN_INPUT_OPTIONS = [
  // { label: "auto", value: "auto", numSeconds: null },
  {value: TimeBin.CENTURY, label: 'bin_century', numSeconds: 3153600000},
  {value: TimeBin.DECADE, label: 'bin_decade', numSeconds: 315360000},
  {value: TimeBin.YEAR, label: 'bin_year', numSeconds: 31536000},
  {value: TimeBin.QUARTER, label: 'bin_quarter', numSeconds: 10368000},
  {value: TimeBin.MONTH, label: 'bin_month', numSeconds: 2592000},
  {value: TimeBin.WEEK, label: 'bin_week', numSeconds: 604800},
  {value: TimeBin.DAY, label: 'bin_day', numSeconds: 86400},
  {value: TimeBin.HOUR, label: 'bin_hour', numSeconds: 3600},
  {value: TimeBin.MINUTE, label: 'bin_minute', numSeconds: 60},
  {value: TimeBin.SECOND, label: 'bin_second', numSeconds: 1},
  {value: TimeBin.MILLISECONDS, label: 'bin_millisecond', numSeconds: 0.001},
];

export const getNumSeconds = (value: string) =>
  (
    TIME_BIN_INPUT_OPTIONS.find((item: any) => item.value === value) ||
    TIME_BIN_INPUT_OPTIONS[Math.floor(TIME_BIN_INPUT_OPTIONS.length / 2)]
  ).numSeconds;

export const EXTRACT_INPUT_OPTIONS = [
  {label: 'ext_century', value: 'century', numSeconds: 3153600000},
  {label: 'ext_decade', value: 'decade', numSeconds: 315360000},
  {label: 'ext_year', value: 'year', numSeconds: 31536000},
  {
    label: 'ext_quarter',
    value: 'quarter',
    numSeconds: 10368000,
    min: 1,
    max: 4,
  },
  {label: 'ext_month', value: 'month', numSeconds: 2592000, min: 1, max: 12},
  {
    label: 'ext_day_of_month',
    value: 'day',
    numSeconds: 86400,
    min: 1,
    max: 31,
  },
  {
    label: 'ext_day_of_week',
    value: 'isodow',
    numSeconds: 86400,
    min: 1,
    max: 7,
  },
  {label: 'ext_hour', value: 'hour', numSeconds: 3600, min: 0, max: 24},
  {label: 'ext_minute', value: 'minute', numSeconds: 60, min: 0, max: 60},
  {label: 'ext_second', value: 'second', numSeconds: 1, min: 0, max: 60},
  // { label: "ext_millisecond", value: "millisecond", numSeconds: 0.001 }
];
