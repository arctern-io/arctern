import {monthDayGetter, getNumSeconds, TIME_BIN_INPUT_OPTIONS, TimeBin} from './Time';

test('Time monthDayGetter', () => {
  expect(monthDayGetter(22)).toEqual('22nd');
  expect(monthDayGetter(31)).toEqual('31st');
  expect(monthDayGetter(30)).toEqual('30th');
  expect(monthDayGetter(23)).toEqual('23rd');
});

test('Time getNumSeconds', () => {
  const hourSeconds = TIME_BIN_INPUT_OPTIONS.find((item: any) => item.value === TimeBin.HOUR)!
    .numSeconds;
  expect(getNumSeconds(TimeBin.HOUR)).toEqual(hourSeconds);
  const defaultSeconds =
    TIME_BIN_INPUT_OPTIONS[Math.floor(TIME_BIN_INPUT_OPTIONS.length / 2)].numSeconds;
  expect(getNumSeconds('get default')).toEqual(defaultSeconds);
});
