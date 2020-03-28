import parseFilter from './parseFilter';

test('parse filter', () => {
  expect(parseFilter({}, {} as any)).toStrictEqual({});

  expect(
    parseFilter(
      {},
      {
        type: 'filter',
        expr: 'where a > 100',
      }
    )
  ).toStrictEqual({where: ['(where a > 100)']});

  expect(
    parseFilter(
      {},
      {
        type: 'filter',
        expr: {type: '=', left: 'car', right: '2'},
      }
    )
  ).toStrictEqual({where: [`(car = '2')`]});

  expect(
    parseFilter(
      {},
      {
        type: 'filter',
        expr: {
          type: 'and',
          left: {type: 'not', expr: {type: 'in', set: [1, 2, 3, 4], expr: 'col'}},
          right: {type: 'coalesce', values: [1, 2, null, 4]},
        },
      }
    )
  ).toStrictEqual({where: [`((NOT(col IN (1, 2, 3, 4)) AND COALESCE(1, 2, NULL, 4)))`]});
});
