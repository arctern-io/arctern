import {getInitLayout, applyUsedLayout} from './Layout';
import {WIDGET} from './Consts';

test('getInitLayout', () => {
  const layout = getInitLayout([]);
  expect(layout).toStrictEqual({
    x: 0,
    y: 0,
    w: 10,
    h: 10,
    static: false,
    minW: 3,
    minH: 1,
  });
});

test('getInitLayout', () => {
  const layout = getInitLayout([
    {x: 0, y: 0, w: 10, h: 10, i: '123', static: false},
    {x: 10, y: 0, w: 10, h: 10, i: '1', static: false},
  ]);
  expect(layout).toStrictEqual({x: 20, y: 0, w: 10, h: 10, static: false, minW: 3, minH: 1});
});

test('getInitLayout', () => {
  const layout = getInitLayout([
    {x: 0, y: 0, w: 10, h: 10, i: '123', static: false, minW: 3, minH: 1},
    {x: 10, y: 0, w: 10, h: 10, i: '1', static: false, minW: 3, minH: 1},
    {x: 20, y: 0, w: 10, h: 10, i: '1', static: false, minW: 3, minH: 1},
    {x: 0, y: 10, w: 10, h: 10, i: '1', static: false, minW: 3, minH: 1},
  ]);
  expect(layout).toStrictEqual({x: 10, y: 10, w: 10, h: 10, static: false, minW: 3, minH: 1});
});

test('applyUsedLayout', () => {
  const configs = [
    {
      id: 'line',
      type: WIDGET.LINECHART,
      title: 'line',
      source: 'test',
      dimensions: [],
      measures: [],
      filter: {},
      selfFilter: {},
      isServerRender: false,
      layout: {x: 5, y: 5, w: 5, h: 5, i: 'line', static: false, minH: 1, minW: 3},
    },
    {
      id: 'Scatter',
      type: WIDGET.SCATTERCHART,
      title: 'Scatter',
      source: 'test',
      dimensions: [],
      measures: [],
      filter: {},
      selfFilter: {},
      isServerRender: false,
      layout: {x: 5, y: 5, w: 5, h: 5, i: 'Scatter', static: false, minH: 1, minW: 3},
    },
    {
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
    },
    {
      id: 'Number',
      type: WIDGET.NUMBERCHART,
      title: 'Number',
      source: 'test',
      dimensions: [],
      measures: [],
      filter: {},
      selfFilter: {},
      isServerRender: false,
      layout: {x: 5, y: 5, w: 5, h: 5, i: 'Number', static: false, minH: 1, minW: 3},
    },
    {
      id: 'Number',
      type: WIDGET.NUMBERCHART,
      title: 'Number',
      source: 'test',
      dimensions: [],
      measures: [],
      filter: {},
      selfFilter: {},
      isServerRender: false,
      layout: {x: 5, y: 5, w: 5, h: 5, i: 'Number', static: false, minH: 1, minW: 3},
    },
    {
      id: 'Number',
      type: WIDGET.NUMBERCHART,
      title: 'Number',
      source: 'test',
      dimensions: [],
      measures: [],
      filter: {},
      selfFilter: {},
      isServerRender: false,
      layout: {x: 5, y: 5, w: 5, h: 5, i: 'Number', static: false, minH: 1, minW: 3},
    },
    {
      id: 'Number',
      type: WIDGET.NUMBERCHART,
      title: 'Number',
      source: 'test',
      dimensions: [],
      measures: [],
      filter: {},
      selfFilter: {},
      isServerRender: false,
      layout: {x: 5, y: 5, w: 5, h: 5, i: 'Number', static: false, minH: 1, minW: 3},
    },
    {
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
    },
  ];
  const newConfigs = applyUsedLayout(configs, '_9avg');
  const newConfigs1 = applyUsedLayout(configs, '_4211');
  const newConfigs2 = applyUsedLayout(configs, '_1124');
  const newConfigs3 = applyUsedLayout(configs, '_timelineTop');
  const newConfigs4 = applyUsedLayout(configs, '_mapdDashboard');

  expect(newConfigs[1]).toStrictEqual({
    id: 'Scatter',
    type: WIDGET.SCATTERCHART,
    title: 'Scatter',
    source: 'test',
    dimensions: [],
    measures: [],
    filter: {},
    selfFilter: {},
    isServerRender: false,
    layout: {x: 10, y: 0, w: 10, h: 10, i: 'Scatter', static: false, minH: 1, minW: 3},
  });
  expect(newConfigs1[0]).toStrictEqual({
    id: 'PointMap',
    type: WIDGET.POINTMAP,
    title: 'PointMap',
    source: 'test',
    dimensions: [],
    measures: [],
    filter: {},
    selfFilter: {},
    isServerRender: false,
    layout: {x: 0, y: 0, w: 15, h: 30, i: 'PointMap', static: false, minH: 1, minW: 3},
  });
  expect(newConfigs2[3]).toStrictEqual({
    id: 'PointMap',
    type: WIDGET.POINTMAP,
    title: 'PointMap',
    source: 'test',
    dimensions: [],
    measures: [],
    filter: {},
    selfFilter: {},
    isServerRender: false,
    layout: {x: 15, y: 0, w: 15, h: 30, i: 'PointMap', static: false, minH: 1, minW: 3},
  });
  expect(newConfigs3[0]).toStrictEqual({
    id: 'line',
    type: WIDGET.LINECHART,
    title: 'line',
    source: 'test',
    dimensions: [],
    measures: [],
    filter: {},
    selfFilter: {},
    isServerRender: false,
    layout: {x: 0, y: 0, w: 30, h: 5, i: 'line', static: false, minH: 1, minW: 3},
  });
  expect(newConfigs4[0]).toStrictEqual({
    id: 'PointMap',
    type: WIDGET.POINTMAP,
    title: 'PointMap',
    source: 'test',
    dimensions: [],
    measures: [],
    filter: {},
    selfFilter: {},
    isServerRender: false,
    layout: {x: 0, y: 0, w: 12, h: 30, i: 'PointMap', static: false, minH: 1, minW: 3},
  });
});
