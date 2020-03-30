import parseSource from './parseSource';

test('parse Source', () => {
  expect(parseSource({}, {} as any)).toStrictEqual({});

  expect(
    parseSource(
      {},
      {
        type: 'source',
        source: [
          {
            type: 'scan',
            table: 'map',
          },
        ],
      }
    )
  ).toStrictEqual({from: 'map'});
  expect(
    parseSource(
      {},
      {
        type: 'source',
        source: [
          {
            type: 'scan',
            table: 'cars',
          },
          {
            type: 'scan',
            table: 'tesla',
          },
          {
            type: 'join',
            as: 'table1',
          },
          {
            type: 'scan',
            table: 'ford',
          },
          {
            type: 'join.right',
            as: 'table2',
          },
          {
            type: 'scan',
            table: 'benz',
          },
          {
            type: 'join.left',
            as: 'table3',
          },
        ],
      }
    )
  ).toStrictEqual({
    from: `cars JOIN tesla AS table1 RIGHT JOIN ford AS table2 LEFT JOIN benz AS table3`,
  });

  expect(
    parseSource(
      {},
      {
        type: 'source',
        source: [
          {
            type: 'scan',
            table: 'cars',
          },
          {
            type: 'scan',
            table: 'zipcode',
          },
          {
            type: 'join.inner',
            as: 'table1',
          },
          {
            type: 'root',
            source: 'cars',
            transform: [
              {
                type: 'aggregate',
                groupby: ['dest_city'],
                fields: ['depdelay'],
                ops: ['average'],
                as: ['val'],
              },
            ],
          } as any,
          {
            type: 'join.left',
            as: 'table2',
          },
        ],
      }
    )
  ).toStrictEqual({
    from:
      'cars INNER JOIN zipcode AS table1 LEFT JOIN (SELECT dest_city, avg(depdelay) AS val FROM cars GROUP BY dest_city) AS table2',
  });
});
