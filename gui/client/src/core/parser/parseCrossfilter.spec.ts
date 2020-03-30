import parseCrossfilter from './parseCrossfilter';

test('parse crossfilter', () => {
  expect(parseCrossfilter({}, {} as any)).toStrictEqual({});

  expect(
    parseCrossfilter(
      {
        unresolved: {
          signal: 'test',
        },
      } as any,
      {
        type: 'crossfilter',
        signal: 'xxx',
        filter: {
          ['my']: {
            type: 'filter',
            expr: 'where a > 100',
          },
        },
      }
    )
  ).toStrictEqual({
    unresolved: {
      signal: 'test',
    },
  });

  expect(
    parseCrossfilter(
      {
        unresolved: {
          crossfilterSignal: {
            ignore: ['my'],
          },
        } as any,
      },
      {
        type: 'crossfilter',
        signal: 'crossfilterSignal',
        filter: {
          ['my']: {
            type: 'filter',
            expr: 'where a > 100',
          },
        },
      }
    )
  ).toStrictEqual({
    unresolved: {
      crossfilterSignal: {
        ignore: ['my'],
      },
    },
  });

  expect(
    parseCrossfilter(
      {
        unresolved: {
          test: {
            ignore: ['mytest', 'abc'],
          },
        } as any,
      },
      {
        type: 'crossfilter',
        signal: 'test',
        filter: {
          ['mytest']: {
            type: 'filter',
            expr: 'where a > 100',
          },
          ['abc']: {
            type: 'filter',
            expr: 'where a > 111',
          },
          ['bde']: {
            type: 'filter',
            expr: 'where a > 38',
          },
        },
      }
    )
  ).toStrictEqual({
    where: [`(where a > 38)`],
    unresolved: {
      test: {
        ignore: ['mytest', 'abc'],
      },
    },
  });
});
