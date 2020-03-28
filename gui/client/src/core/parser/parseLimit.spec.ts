import parseLimit from './parseLimit';

test('parse limit', () => {
  expect(parseLimit({}, {})).toStrictEqual({})

  expect(
    parseLimit(
      {},
      {
        type: 'limit',
        limit: 1,
        offset: 100,
      }
    )
  ).toStrictEqual({limit: 1, offset: 100});
});
