import Parser from '.';

test('Parser', () => {
  Parser.registerExpression('custom', () => 'TEST');
  Parser.registerTransform('custom', () => ({trans: 1}));

  expect(Parser.parseExpression({type: 'custom'})).toBe('TEST');
  expect(Parser.parseTransform({}, {type: 'custom'})).toStrictEqual({trans: 1});

  expect(Parser.parseExpression({type: '=', left: '1', right: '2'})).toBe(`1 = '2'`);
  expect(
    Parser.parseTransform(
      {},
      {
        type: 'aggregate',
        fields: ['*'],
        ops: ['count'],
        as: ['series_1'],
        groupby: {
          type: 'bin',
          field: 'binCol',
          extent: [-21474830, 3950611.6],
          maxbins: 12,
          as: 'binColAs',
        },
      }
    )
  ).toStrictEqual({
    groupby: ['binColAs'],
    having: [
      `((CASE WHEN binCol >= 3950611.6 THEN 11 else cast((cast(binCol AS float) - -21474830) * 4.719682036909046e-7 AS int) end) >= 0 AND (CASE WHEN binCol >= 3950611.6 THEN 11 else cast((cast(binCol AS float) - -21474830) * 4.719682036909046e-7 AS int) end) < 12 OR (CASE WHEN binCol >= 3950611.6 THEN 11 else cast((cast(binCol AS float) - -21474830) * 4.719682036909046e-7 AS int) end) IS NULL)`,
    ],
    select: [
      `CASE WHEN binCol >= 3950611.6 THEN 11 else cast((cast(binCol AS float) - -21474830) * 4.719682036909046e-7 AS int) end AS binColAs`,
      `count(*) AS series_1`,
    ],
    where: [`((binCol >= -21474830 AND binCol <= 3950611.6) OR (binCol IS NULL))`],
  });
});
