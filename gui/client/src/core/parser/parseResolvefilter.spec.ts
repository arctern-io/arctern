import parseResolvefilter from './parseResolvefilter';

test('parse resolve filter', () => {
  expect(parseResolvefilter({}, {} as any)).toStrictEqual({});

  expect(
    parseResolvefilter(
      {},
      {
        type: 'resolvefilter',
        filter: {
          signal: 'xDimension',
        },
      }
    )
  ).toStrictEqual({
    unresolved: {
      xDimension: {
        filter: {
          signal: 'xDimension',
        },
        type: 'resolvefilter',
      },
    },
  });
});
