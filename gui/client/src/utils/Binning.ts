import {TIME_BIN_INPUT_OPTIONS, EXTRACT_INPUT_OPTIONS} from './Time';

const minBinGrp = 2;
const maxBinGrp = 1000;
const QUARTERS = [
  [1, 2, 3],
  [4, 5, 6],
  [7, 8, 9],
  [10, 11, 12],
];

export const parseDate = (date: Date) => {
  return {
    year: date.getFullYear(),
    month: date.getMonth() + 1,
    date: date.getDate(),
    day: date.getDay(),
    hour: date.getHours(),
    minute: date.getMinutes(),
    second: date.getSeconds(),
    // millisecond: date.getMilliseconds(),
  };
};

export const getValidTimeBinOpts = (min: any, max: any, isExtract: boolean) => {
  const [dateMin, dateMax]: Date[] = [new Date(min), new Date(max)];
  const GAP_IN_MILLISECS = Math.floor(Date.parse(max) - Date.parse(min));
  const {
    year: minYear,
    month: minMonth,
    date: minDate,
    day: minDay,
    hour: minHour,
    minute: minMinute,
    second: minSecond,
    // millisecond: minMillisecond,
  } = parseDate(dateMin);

  const {
    year: maxYear,
    month: maxMonth,
    date: maxDate,
    day: maxDay,
    hour: maxHour,
    minute: maxMinute,
    second: maxSecond,
    // millisecond: maxMillisecond,
  } = parseDate(dateMax);

  // For extract , if the bigger time unit is not same, the smaller time unit will return anyway.
  // Example: if quarter is not same => month & day & isodow & hour &minute & second will all return
  let isNotSameTime: boolean = false,
    temp = false;

  return isExtract
    ? EXTRACT_INPUT_OPTIONS.filter((option: any) => {
        switch (option.value) {
          case 'century':
          case 'decade':
          case 'year':
            return false;
          case 'quarter':
            let sameYear = minYear === maxYear,
              sameQuater = QUARTERS.some(
                quarter => quarter.includes(minMonth) && quarter.includes(maxMonth)
              );
            temp = !(sameYear && sameQuater);
            break;
          case 'month':
            temp = minMonth !== maxMonth;
            break;
          case 'day':
            temp = minDate !== maxDate;
            break;
          case 'isodow':
            temp = minDay !== maxDay;
            break;
          case 'hour':
            temp = minHour !== maxHour;
            break;
          case 'minute':
            temp = minMinute !== maxMinute;
            break;
          case 'second':
            temp = minSecond !== maxSecond;
            break;
          // case 'millisecond':
          //   temp = minMillisecond !== maxMillisecond;
          //   break;
          default:
            return false;
        }
        isNotSameTime = isNotSameTime || temp;
        return isNotSameTime;
      })
    : TIME_BIN_INPUT_OPTIONS.filter((option: any) => {
        let numOfGroup = Math.floor(GAP_IN_MILLISECS / option.numSeconds / 1000);
        return numOfGroup >= minBinGrp && numOfGroup <= maxBinGrp;
      });
};
