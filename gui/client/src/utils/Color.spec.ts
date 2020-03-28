import {genColorGetter, getValidColorKey, isGradientType} from './Colors';
import {WIDGET} from './Consts';

test('genColorGetter', () => {
  const solidColorConfig = {
    id: 'PointMap',
    type: WIDGET.POINTMAP,
    title: 'PointMap',
    source: 'test',
    dimensions: [],
    measures: [],
    filter: {},
    selfFilter: {},
    isServerRender: false,
    layout: {x: 5, y: 5, w: 5, h: 5, i: 'PointMap', static: false, minH: 1, minW: 3},
    colorKey: '#FFDB5C',
  };
  const gradientColorConfig = {
    id: 'PointMap',
    type: WIDGET.POINTMAP,
    title: 'PointMap',
    source: 'test',
    dimensions: [],
    measures: [],
    filter: {},
    selfFilter: {},
    isServerRender: false,
    layout: {x: 5, y: 5, w: 5, h: 5, i: 'PointMap', static: false, minH: 1, minW: 3},
    colorKey: 'purple_to_yellow',
  };
  const ordinalColorConfig = {
    id: 'Pie',
    type: WIDGET.PIECHART,
    title: 'Pie',
    source: 'test',
    dimensions: [],
    measures: [],
    filter: {},
    selfFilter: {},
    isServerRender: false,
    layout: {x: 5, y: 5, w: 5, h: 5, i: 'Pie', static: false, minH: 1, minW: 3},
    colorKey: 'ordinal3',
  };

  const solidRes = genColorGetter(solidColorConfig);
  const gradientRes = genColorGetter(gradientColorConfig);
  const ordinalRes = genColorGetter(ordinalColorConfig);

  expect(typeof solidRes).toBe('function');
  expect(typeof gradientRes).toBe('function');
  expect(typeof ordinalRes).toBe('function');
});

test('getValidColorKey', () => {
  const res1 = getValidColorKey('#FFDB5C', ['solid', 'ordinal']);
  const res2 = getValidColorKey('purple_to_yellow', ['solid', 'ordinal']);
  const res3 = getValidColorKey('ordinal3', ['solid', 'ordinal']);
  const res4 = getValidColorKey('purple_to_yellow', ['solid', 'gradient']);
  const res5 = getValidColorKey('lallalalalla', ['solid', 'ordinal', 'gradient']);
  const res6 = getValidColorKey('123123123123', ['solid', 'ordinal', 'gradient']);
  const res7 = getValidColorKey('undefined', ['solid', 'ordinal', 'gradient']);

  expect(res1).toBe('#FFDB5C');
  expect(res2).toBe('#37A2DA');
  expect(res3).toBe('ordinal3');
  expect(res4).toBe('purple_to_yellow');
  expect(res5).toBe('#37A2DA');
  expect(res6).toBe('#37A2DA');
  expect(res7).toBe('#37A2DA');
});

test('isGradientType', () => {
  const res1 = isGradientType('#FFDB5C');
  const res2 = isGradientType('purple_to_yellow');
  const res3 = isGradientType('ordinal3');
  const res4 = isGradientType('false');
  const res5 = isGradientType('lallalallla');
  const res6 = isGradientType('123123123123');

  expect(res1).toBe(false);
  expect(res2).toBe(true);
  expect(res3).toBe(false);
  expect(res4).toBe(false);
  expect(res5).toBe(false);
  expect(res6).toBe(false);
});
