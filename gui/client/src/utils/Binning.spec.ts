import {getValidTimeBinOpts, parseDate} from './Binning';

test('Binning parseDate', () => {
  const myDate = new Date('2019-01-01 11:11:11');
  const {year, month, day, date, hour, minute, second} = parseDate(myDate);
  expect(year).toEqual(2019);
  expect(month).toEqual(1);
  expect(date).toEqual(1);
  expect(hour).toEqual(11);
  expect(minute).toEqual(11);
  expect(second).toEqual(11);
  expect(day).toEqual(2); // weekday
});
test('Binning getValidTimeBinOpts Same DAY Diff Min', () => {
  let minDate = 'Fri May 01 00:01:45 2009';
  let maxDate = 'Fri May 01 00:44:00 2009';
  let timeBin = getValidTimeBinOpts(minDate, maxDate, false);
  expect(timeBin).toEqual([{value: 'minute', label: 'bin_minute', numSeconds: 60}]);

  let extractTimeBin = getValidTimeBinOpts(minDate, maxDate, true);
  expect(extractTimeBin).toEqual([
    {
      label: 'ext_minute',
      value: 'minute',
      numSeconds: 60,
      min: 0,
      max: 60,
    },
    {
      label: 'ext_second',
      value: 'second',
      numSeconds: 1,
      min: 0,
      max: 60,
    },
  ]);
});

test('Binning getValidTimeBinOpts Diff Quarter', () => {
  const minDate = 'Sat Jan 12 12:40:00 2019';
  const maxDate = 'Fri Apr 12 12:40:00 2019';
  const timeBin = getValidTimeBinOpts(minDate, maxDate, false);
  expect(timeBin).toEqual([
    {value: 'month', label: 'bin_month', numSeconds: 2592000},
    {value: 'week', label: 'bin_week', numSeconds: 604800},
    {value: 'day', label: 'bin_day', numSeconds: 86400},
  ]);

  const extractTimeBin = getValidTimeBinOpts(minDate, maxDate, true);
  expect(extractTimeBin).toEqual([
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
  ]);
});
