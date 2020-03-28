import parseHaving from './parseHaving';

test('parse limit', () => {
  expect(parseHaving({}, {})).toStrictEqual({});

  expect(
    parseHaving(
      {},
      {
        type: 'having',
        expr: {type: 'like', left: 'col', right: 'infini'},
      }
    )
  ).toStrictEqual({having: [`col LIKE '%infini%'`]});

  expect(
    parseHaving(
      {},
      {
        type: 'having',
        expr: {
          type: 'or',
          left: {type: 'not', expr: {type: 'in', set: ['1', '2', '3', '4'], expr: 'col'}},
          right: {type: 'coalesce', values: ['1', '2', null, '4']},
        },
      }
    )
  ).toStrictEqual({having: [`(NOT(col IN ('1', '2', '3', '4')) OR COALESCE(1, 2, NULL, 4))`]});
});
