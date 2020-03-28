import {
  AliasExpression,
  Expression,
  AvgExpression,
  MaxExpression,
  MinExpression,
  SumExpression,
  CountExpression,
  ExtractExpression,
  DateTruncExpression,
  BooleanExpression,
  CaseExpression,
  NotExpression,
  BetweenExpression,
  ExtractUnits,
  DateTruncUnits,
} from '../types';

export const alias = (as: string, expr: string | Expression): AliasExpression => ({
  expr,
  as,
});

export const avg = (alias: string, field: string): AvgExpression => ({
  type: 'average',
  field,
  as: alias,
});

export const min = (alias: string, field: string): MinExpression => ({
  type: 'min',
  field,
  as: alias,
});

export const max = (alias: string, field: string): MaxExpression => ({
  type: 'max',
  field,
  as: alias,
});

export const sum = (alias: string, field: string): SumExpression => ({
  type: 'sum',
  field,
  as: alias,
});

export const count = (distinct: boolean, alias: string, field: string): CountExpression => ({
  type: 'count',
  distinct,
  approx: false,
  field,
  as: alias,
});

export const approxCount = (distinct: boolean, alias: string, field: string): CountExpression => ({
  type: 'count',
  distinct,
  approx: true,
  field,
  as: alias,
});

export const countStar = (alias: string): CountExpression => ({
  type: 'count',
  distinct: false,
  approx: false,
  field: '*',
  as: alias,
});

export const extract = (unit: ExtractUnits, field: string): ExtractExpression => ({
  type: 'extract',
  unit,
  field,
});

export const dateTrunc = (unit: DateTruncUnits, field: string): DateTruncExpression => ({
  type: 'date_trunc',
  unit,
  field,
});

export const inExpr = (expr: string, set: string | any | Array<string | number>) => ({
  type: 'in',
  expr,
  set,
});

export const not = (expr: string | BooleanExpression): NotExpression => ({
  type: 'not',
  expr,
});

export const caseExpr = (
  cond: Array<[BooleanExpression | string, string]>,
  end: string
): CaseExpression => ({
  type: 'case',
  cond,
  else: end,
});

export const between = (field: string, range: Array<number | string>): BetweenExpression => ({
  type: 'between',
  field,
  left: range[0],
  right: range[1],
});
