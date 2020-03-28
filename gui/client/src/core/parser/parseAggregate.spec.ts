import parseAggregate from './parseAggregate';

test('parseAggregate', () => {
  expect(parseAggregate({}, {} as any)).toStrictEqual({});

  expect(
    parseAggregate(
      {},
      {
        type: 'aggregate',
        groupby: 'a',
        ops: ['min'],
        as: ['asA'],
        fields: ['c'],
      }
    )
  ).toStrictEqual({
    select: ['a', 'min(c) AS asA'],
    groupby: ['a'],
  });

  expect(
    parseAggregate(
      {},
      {
        type: 'aggregate',
        groupby: ['a', 'b'],
        ops: ['min', 'max'],
        as: ['asA', 'asB'],
        fields: ['c', 'd'],
      }
    )
  ).toStrictEqual({
    select: ['a', 'b', 'min(c) AS asA', 'max(d) AS asB'],
    groupby: ['a', 'b'],
  });

  expect(
    parseAggregate(
      {},
      {
        type: 'aggregate',
        groupby: [{type: 'project', expr: 'projectCol', as: 'ppp'}, 'b'],
        ops: ['min', 'max'],
        as: ['asA', 'asB'],
        fields: ['c', 'd'],
      }
    )
  ).toStrictEqual({
    select: ['projectCol AS ppp', 'b', 'min(c) AS asA', 'max(d) AS asB'],
    groupby: ['ppp', 'b'],
  });

  expect(
    parseAggregate(
      {},
      {
        type: 'aggregate',
        groupby: [
          {type: 'project', expr: 'projectCol', as: 'ppp'},
          {
            type: 'bin',
            field: 'car',
            as: 'mycar',
            extent: [0, 100],
            maxbins: 25,
          },
        ],
        ops: ['min', 'max'],
        as: ['asA', 'asB'],
        fields: ['c', 'd'],
      }
    )
  ).toStrictEqual({
    select: [
      'projectCol AS ppp',
      `CASE WHEN car >= 100 THEN 24 else cast((cast(car AS float) - 0) * 0.25 AS int) end AS mycar`,
      'min(c) AS asA',
      'max(d) AS asB',
    ],
    having: [
      `((CASE WHEN car >= 100 THEN 24 else cast((cast(car AS float) - 0) * 0.25 AS int) end) >= 0 AND (CASE WHEN car >= 100 THEN 24 else cast((cast(car AS float) - 0) * 0.25 AS int) end) < 25 OR (CASE WHEN car >= 100 THEN 24 else cast((cast(car AS float) - 0) * 0.25 AS int) end) IS NULL)`,
    ],
    groupby: ['ppp', 'mycar'],
    where: [`((car >= 0 AND car <= 100) OR (car IS NULL))`],
  });
});
