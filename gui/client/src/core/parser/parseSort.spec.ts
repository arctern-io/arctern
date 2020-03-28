import parseSort from './parseSort';

test('parse sort', () => {
  expect(parseSort({}, {} as any)).toStrictEqual({})

  expect(
    parseSort(
      {},
      {
        type: 'sort',
        field: ['col0', 'col1'],
        order: ['ascending', 'descending'],
      }
    )
  ).toStrictEqual({orderby: ['col0 ASC', 'col1 DESC']});

  expect(
    parseSort(
      {},
      {
        type: 'sort',
        field: ['col0', 'col1'],
        order: ['desc', 'asc'],
      }
    )
  ).toStrictEqual({orderby: ['col0 DESC', 'col1 ASC']});

  expect(
    parseSort(
      {},
      {
        type: 'sort',
        field: ['col0', 'col1', 'col3'],
        order: ['desc', 'descending'],
      }
    )
  ).toStrictEqual({orderby: ['col0 DESC', 'col1 DESC', 'col3 ASC']});
});
