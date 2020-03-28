import parseProject from './parseProject';

test('parse project', () => {
  expect(parseProject({}, {} as any)).toStrictEqual({});

  expect(
    parseProject(
      {},
      {
        type: 'project',
        expr: 'col',
        as: 'ass1',
      }
    )
  ).toStrictEqual({select: ['col AS ass1']});

  expect(
    parseProject(
      {},
      {
        type: 'project',
        expr: {type: 'avg', field: 'col'},
        as: 'ass1',
      }
    )
  ).toStrictEqual({select: ['avg(col) AS ass1']});
});
