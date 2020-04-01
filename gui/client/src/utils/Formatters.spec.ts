import {DATE_FORMAT, NUM_FORMAT, autoNumDimensionFormat, formatterGetter} from './Formatters';
import {Dimension} from '../types';

test('autoNumDimensionFormat', () => {
  const formatter1 = autoNumDimensionFormat({extent: [1, 100]});
  const formatter2 = autoNumDimensionFormat({extent: [1, 10]});
  const formatter3 = autoNumDimensionFormat({extent: [-100, 1000000000]});

  expect(formatter1(100)).toBe('100.0');
  expect(formatter2(10)).toBe('10.0');
  expect(formatter3(100)).toBe('100.00');
  expect(formatter3(10000)).toBe('10k');
});

test('formatterGetter', () => {
  DATE_FORMAT.forEach((item: any) => {
    const dateFormatter = formatterGetter({
      format: item.value,
      type: 'timestamp',
      timeBin: 'day',
      isBinned: true,
    } as Dimension);
    const res = dateFormatter('Wed, 08 Jan 2020 07:29:19 GMT');
    switch (item.value) {
      case '%y':
        expect(res).toBe('20');
        break;
      case '%m/%d/%y':
        expect(res).toBe('08/01/20');
        break;
      case '%y-%m-%d':
        expect(res).toBe('20-01-08');
        break;
      case '%b':
        expect(res).toBe('Jan');
        break;
      case '%b %d':
        expect(res).toBe('Jan 08');
        break;
      case '%a %d':
        expect(res).toBe('Wed 08');
        break;
      case '%I%p, %d':
        expect(res).toBe('03PM, 08');
        break;
      case '%I%p':
        expect(res).toBe('03PM');
        break;
      case '%X':
        expect(res).toBe('3:29:19 PM');
        break;
      case '%H:%m:%S':
        expect(res).toBe('15:01:19');
        break;
      case 'auto':
        expect(res).toBe('1/8/2020 15:29');
        break;
      default:
        break;
    }
  });

  NUM_FORMAT.forEach((item: any) => {
    const numberFormatter = formatterGetter({format: item.value, type: 'float8'});
    const res = numberFormatter(38.976);
    switch (item.value) {
      case 'auto':
        expect(res).toBe(`38.98`);
        break;
      case ',.2f':
        expect(res).toBe(`38.98`);
        break;
      case ',.0f':
        expect(res).toBe(`39`);
        break;
      case '.2s':
        expect(res).toBe(`39`);
        break;
      case ',.2%':
        expect(res).toBe(`3,897.60%`);
        break;
      case '-$.2f':
        expect(res).toBe(`$38.98`);
        break;
      default:
        break;
    }
  });
});
