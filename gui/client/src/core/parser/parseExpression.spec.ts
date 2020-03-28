import parseExpression from './parseExpression';

test('=,<>,<,>,<=,>=', () => {
  expect(parseExpression({type: '=', left: '1', right: '2'})).toBe(`1 = '2'`);
  expect(parseExpression({type: '<>', left: '1', right: '2'})).toBe(`1 <> '2'`);
  expect(parseExpression({type: '<', left: '1', right: '2'})).toBe(`1 < '2'`);
  expect(parseExpression({type: '>=', left: '1', right: '2'})).toBe(`1 >= '2'`);
  expect(parseExpression({type: '<=', left: '1', right: '2'})).toBe(`1 <= '2'`);
});

test('string', () => {
  expect(parseExpression('test')).toBe(`test`);
});

test('null', () => {
  expect(parseExpression(null)).toBe(`NULL`);
});

test('between, not between', () => {
  expect(parseExpression({type: 'between', field: 'col', left: '1', right: '2'})).toBe(
    `col BETWEEN '1' AND '2'`
  );
  expect(parseExpression({type: 'not between', field: 'col', left: '1', right: '2'})).toBe(
    `col NOT BETWEEN '1' AND '2'`
  );
});

test('is null, is not null', () => {
  expect(parseExpression({type: 'is null', field: 'col'})).toBe(`col IS NULL`);
  expect(parseExpression({type: 'is not null', field: 'col'})).toBe(`col IS NOT NULL`);
});

test('ilike, like, not like', () => {
  expect(parseExpression({type: 'ilike', left: 'col', right: 'infini'})).toBe(
    `col ILIKE '%infini%'`
  );
  expect(parseExpression({type: 'like', left: 'col', right: 'infini'})).toBe(`col LIKE '%infini%'`);
  expect(parseExpression({type: 'not like', left: 'col', right: 'infini'})).toBe(
    `col NOT LIKE '%infini%'`
  );
});

test('coalesce', () => {
  expect(parseExpression({type: 'coalesce', values: [1, 2, null, 4]})).toBe(
    `COALESCE(1, 2, NULL, 4)`
  );
  expect(
    parseExpression({
      type: 'coalesce',
      values: [{type: 'avg', field: 'col'}, 2, null, 4],
    })
  ).toBe(`COALESCE(avg(col), 2, NULL, 4)`);
});

test('in, not in', () => {
  expect(parseExpression({type: 'in', set: [1, 2, 3, 4], expr: 'col'})).toBe(`col IN (1, 2, 3, 4)`);
  expect(parseExpression({type: 'in', set: ['1', '2', '3', '4'], expr: 'col'})).toBe(
    `col IN ('1', '2', '3', '4')`
  );
  expect(parseExpression({type: 'not in', set: [1, 2, 3, 4], expr: 'col'})).toBe(
    `col NOT IN (1, 2, 3, 4)`
  );
  expect(parseExpression({type: 'not in', set: ['1', '2', '3', '4'], expr: 'col'})).toBe(
    `col NOT IN ('1', '2', '3', '4')`
  );
});

test('not', () => {
  expect(parseExpression({type: 'not', expr: {type: 'in', set: [1, 2, 3, 4], expr: 'col'}})).toBe(
    `NOT(col IN (1, 2, 3, 4))`
  );
});

test('and, or', () => {
  expect(parseExpression({type: 'and', left: 'col1', right: 'col2'})).toBe(`(col1 AND col2)`);
  expect(parseExpression({type: 'or', left: 'col1', right: 'col2'})).toBe(`(col1 OR col2)`);

  expect(
    parseExpression({
      type: 'and',
      left: {type: 'not', expr: {type: 'in', set: [1, 2, 3, 4], expr: 'col'}},
      right: {type: 'coalesce', values: [1, 2, null, 4]},
    })
  ).toBe(`(NOT(col IN (1, 2, 3, 4)) AND COALESCE(1, 2, NULL, 4))`);

  expect(
    parseExpression({
      type: 'or',
      left: {type: 'not', expr: {type: 'in', set: [1, 2, 3, 4], expr: 'col'}},
      right: {type: 'coalesce', values: [1, 2, null, 4]},
    })
  ).toBe(`(NOT(col IN (1, 2, 3, 4)) OR COALESCE(1, 2, NULL, 4))`);
});

test('case', () => {
  expect(
    parseExpression({
      type: 'case',
      cond: [[{type: 'coalesce', values: [1, 2, null, 4]}, 1]],
    })
  ).toBe(`CASE WHEN COALESCE(1, 2, NULL, 4) THEN 1 END`);
  expect(
    parseExpression({
      type: 'case',
      cond: [[{type: 'coalesce', values: [1, 2, null, 4]}, 1]],
      else: null,
    })
  ).toBe(`CASE WHEN COALESCE(1, 2, NULL, 4) THEN 1 ELSE NULL END`);

  expect(
    parseExpression({
      type: 'case',
      cond: [[{type: 'coalesce', values: [1, 2, null, 4]}, 1]],
      else: 2,
    })
  ).toBe(`CASE WHEN COALESCE(1, 2, NULL, 4) THEN 1 ELSE '2' END`);
});

test('date_trunc', () => {
  expect(parseExpression({type: 'date_trunc', unit: 'month', field: 'col'})).toBe(
    `date_trunc('month', col)`
  );
});

test('extract', () => {
  expect(parseExpression({type: 'extract', unit: 'month', field: 'col'})).toBe(
    `extract(month from col)`
  );
});

test('count', () => {
  expect(parseExpression({type: 'count', field: 'col'})).toBe(`count(col)`);
  expect(parseExpression({type: 'count', distinct: true, field: 'col'})).toBe(
    `count(distinct col)`
  );
});

test('stddev, stddev_pop, stddev_samp, var_pop, var_samp', () => {
  expect(parseExpression({type: 'stddev', expr: 'col'})).toBe(`stddev(col)`);
  expect(parseExpression({type: 'stddev', expr: {type: 'count', field: 'col'}})).toBe(
    `stddev(count(col))`
  );
  expect(parseExpression({type: 'stddev_pop', expr: 'col'})).toBe(`stddev_pop(col)`);
  expect(parseExpression({type: 'stddev_samp', expr: 'col'})).toBe(`stddev_samp(col)`);
  expect(parseExpression({type: 'var_pop', expr: 'col'})).toBe(`var_pop(col)`);
  expect(parseExpression({type: 'var_samp', expr: 'col'})).toBe(`var_samp(col)`);
});

test('corr, covar_pop, covar_samp', () => {
  expect(parseExpression({type: 'corr', x: 1, y: 2})).toBe(`corr(2, 1)`);
  expect(parseExpression({type: 'covar_pop', x: 1, y: 2})).toBe(`covar_pop(2, 1)`);
  expect(parseExpression({type: 'covar_samp', x: 1, y: 2})).toBe(`covar_samp(2, 1)`);
  expect(parseExpression({type: 'regr_avgx', x: 1, y: 2})).toBe(`regr_avgx(2, 1)`);
  expect(parseExpression({type: 'regr_avgy', x: 1, y: 2})).toBe(`regr_avgy(2, 1)`);
  expect(parseExpression({type: 'regr_count', x: 1, y: 2})).toBe(`regr_count(2, 1)`);
  expect(parseExpression({type: 'regr_intercept', x: 1, y: 2})).toBe(`regr_intercept(2, 1)`);
  expect(parseExpression({type: 'regr_r2', x: 1, y: 2})).toBe(`regr_r2(2, 1)`);
  expect(parseExpression({type: 'regr_slope', x: 1, y: 2})).toBe(`regr_slope(2, 1)`);
  expect(parseExpression({type: 'regr_sxx', x: 1, y: 2})).toBe(`regr_sxx(2, 1)`);
  expect(parseExpression({type: 'regr_sxy', x: 1, y: 2})).toBe(`regr_sxy(2, 1)`);
  expect(parseExpression({type: 'regr_syy', x: 1, y: 2})).toBe(`regr_syy(2, 1)`);
});

test('avg, bit_and, bit_or, bool_and, bool_or, every, max, sum', () => {
  expect(parseExpression({type: 'avg', field: 'col'})).toBe(`avg(col)`);
  expect(parseExpression({type: 'bit_and', field: 'col'})).toBe(`bit_and(col)`);
  expect(parseExpression({type: 'bit_or', field: 'col'})).toBe(`bit_or(col)`);
  expect(parseExpression({type: 'bool_and', field: 'col'})).toBe(`bool_and(col)`);
  expect(parseExpression({type: 'bool_or', field: 'col'})).toBe(`bool_or(col)`);
  expect(parseExpression({type: 'every', field: 'col'})).toBe(`every(col)`);
  expect(parseExpression({type: 'max', field: 'col'})).toBe(`max(col)`);
  expect(parseExpression({type: 'min', field: 'col'})).toBe(`min(col)`);
  expect(parseExpression({type: 'sum', field: 'col'})).toBe(`sum(col)`);
  expect(parseExpression({type: 'average', field: 'col'})).toBe(`avg(col)`);
});

test('project', () => {
  expect(parseExpression({type: 'project', field: 'col'})).toBe(`col`);
});

test('polygon', () => {
  expect(parseExpression({type: 'polygon', x: 1, y: 2, px: 3, py: 4})).toBe(
    `is_in_polygon(1, 2, ARRAY[3], ARRAY[4])`
  );
});

test('gis_mapping_lon', () => {
  expect(
    parseExpression({
      type: 'gis_mapping_lon',
      domainStart: 1,
      domainEnd: 2,
      field: 'col',
      range: 100,
    })
  ).toBe(`gis_discrete_trans_scale_long_epsg_4326_900913 (1::float, 2::float, col, 100)`);
});

test('gis_mapping_lat', () => {
  expect(
    parseExpression({
      type: 'gis_mapping_lat',
      domainStart: 1,
      domainEnd: 2,
      field: 'col',
      range: 100,
    })
  ).toBe(`gis_discrete_trans_scale_lat_epsg_4326_900913 (1::float, 2::float, col, 100)`);
});

test('gis_discrete_trans_scale_w', () => {
  expect(
    parseExpression({type: 'gis_discrete_trans_scale_w', domain: [0, 1], field: 'col', width: 100})
  ).toBe(`gis_discrete_trans_scale(0, 1, 0, 99, col::float)`);
});

test('gis_discrete_trans_scale_h', () => {
  expect(
    parseExpression({type: 'gis_discrete_trans_scale_h', domain: [0, 1], field: 'col', height: 100})
  ).toBe(`gis_discrete_trans_scale(0, 1, 0, 99, col::float)`);
});

test('circle', () => {
  expect(
    parseExpression({
      type: 'circle',
      fromlon: 'colLon',
      fromlat: 'colLat',
      distance: 1000,
      tolon: '123',
      tolat: '2123',
    })
  ).toBe(`is_in_circle(colLon, colLat, 1000, 123, 2123)`);
});
