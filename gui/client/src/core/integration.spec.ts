import {InfiniCollection} from './infini';
import {reducer, toSQL} from './parser/reducer';

function concat(transform: any) {
  return (transforms: any) => transforms.concat(transform);
}

test('Integration Test', () => {
  const collection = new InfiniCollection({
    reducer: reducer,
    stringify: toSQL,
  });
  const xFilterNode1 = collection.create('table1');
  const xFilterNode2 = collection.create('table2');

  const childNode1 = xFilterNode1.add({
    transform: [
      {
        type: 'aggregate',
        fields: ['*'],
        ops: ['count'],
        as: ['col'],
        groupby: {
          type: 'project',
          expr: 'type',
          as: 'as0',
        },
      },
      {
        type: 'sort',
        field: ['col'],
        order: ['ascending'],
      },
      {
        type: 'limit',
        row: 10,
      },
    ],
  } as any);

  const childNode2 = xFilterNode1.add({
    transform: [
      {
        type: 'aggregate',
        fields: ['*'],
        ops: ['count'],
        as: ['col'],
        groupby: {
          type: 'bin',
          field: 'binCol',
          extent: [0, 30],
          maxbins: 30,
          as: 'key0',
        },
      },
    ],
  });

  expect(xFilterNode1.reduceToString()).toEqual('SELECT * FROM table1');
  expect(xFilterNode2.reduceToString()).toEqual('SELECT * FROM table2');

  expect(childNode1.reduceToString()).toEqual(
    'SELECT type AS as0, count(*) AS col FROM table1 GROUP BY as0 ORDER BY col ASC'
  );

  expect(childNode2.reduceToString()).toEqual(
    'SELECT CASE WHEN binCol >= 30 THEN 29 else cast((cast(binCol AS float) - 0) * 1 AS int) end AS key0, count(*) AS col FROM table1 WHERE ((binCol >= 0 AND binCol <= 30) OR (binCol IS NULL)) GROUP BY key0 HAVING ((CASE WHEN binCol >= 30 THEN 29 else cast((cast(binCol AS float) - 0) * 1 AS int) end) >= 0 AND (CASE WHEN binCol >= 30 THEN 29 else cast((cast(binCol AS float) - 0) * 1 AS int) end) < 30 OR (CASE WHEN binCol >= 30 THEN 29 else cast((cast(binCol AS float) - 0) * 1 AS int) end) IS NULL)'
  );

  collection.setTransform(
    concat({
      type: 'filter',
      id: 'test',
      expr: {
        type: 'between',
        field: 'betweenCol',
        left: 22222,
        right: 33333,
      },
    })
  );

  expect(xFilterNode1.reduceToString()).toEqual(
    'SELECT * FROM table1 WHERE (betweenCol BETWEEN 22222 AND 33333)'
  );
  expect(xFilterNode2.reduceToString()).toEqual(
    'SELECT * FROM table2 WHERE (betweenCol BETWEEN 22222 AND 33333)'
  );

  expect(childNode1.reduceToString()).toEqual(
    'SELECT type AS as0, count(*) AS col FROM table1 WHERE (betweenCol BETWEEN 22222 AND 33333) GROUP BY as0 ORDER BY col ASC'
  );

  expect(childNode2.reduceToString()).toEqual(
    'SELECT CASE WHEN binCol >= 30 THEN 29 else cast((cast(binCol AS float) - 0) * 1 AS int) end AS key0, count(*) AS col FROM table1 WHERE ((binCol >= 0 AND binCol <= 30) OR (binCol IS NULL)) AND (betweenCol BETWEEN 22222 AND 33333) GROUP BY key0 HAVING ((CASE WHEN binCol >= 30 THEN 29 else cast((cast(binCol AS float) - 0) * 1 AS int) end) >= 0 AND (CASE WHEN binCol >= 30 THEN 29 else cast((cast(binCol AS float) - 0) * 1 AS int) end) < 30 OR (CASE WHEN binCol >= 30 THEN 29 else cast((cast(binCol AS float) - 0) * 1 AS int) end) IS NULL)'
  );

  xFilterNode1.setTransform((t: any) => {
    t[1] = {
      type: 'crossfilter',
      signal: 'x',
      filter: {
        ['abc']: {
          type: 'filter',
          expr: {
            type: '=',
            left: 'type',
            right: 'cash',
          },
        },
        ['def']: {
          type: 'filter',
          expr: {
            type: 'between',
            field: 'binCol',
            left: 12,
            right: 20,
          },
        },
      },
    };
    return t;
  });

  childNode1.setTransform(
    concat({
      type: 'resolvefilter',
      filter: {signal: 'x'},
      ignore: 'abc',
    })
  );

  childNode2.setTransform(
    concat({
      type: 'resolvefilter',
      filter: {signal: 'x'},
      ignore: 'def',
    })
  );

  expect(childNode1.reduceToString()).toEqual(
    'SELECT type AS as0, count(*) AS col FROM table1 WHERE (betweenCol BETWEEN 22222 AND 33333) AND (binCol BETWEEN 12 AND 20) GROUP BY as0 ORDER BY col ASC'
  );

  expect(childNode2.reduceToString()).toEqual(
    "SELECT CASE WHEN binCol >= 30 THEN 29 else cast((cast(binCol AS float) - 0) * 1 AS int) end AS key0, count(*) AS col FROM table1 WHERE ((binCol >= 0 AND binCol <= 30) OR (binCol IS NULL)) AND (betweenCol BETWEEN 22222 AND 33333) AND (type = 'cash') GROUP BY key0 HAVING ((CASE WHEN binCol >= 30 THEN 29 else cast((cast(binCol AS float) - 0) * 1 AS int) end) >= 0 AND (CASE WHEN binCol >= 30 THEN 29 else cast((cast(binCol AS float) - 0) * 1 AS int) end) < 30 OR (CASE WHEN binCol >= 30 THEN 29 else cast((cast(binCol AS float) - 0) * 1 AS int) end) IS NULL)"
  );
});
