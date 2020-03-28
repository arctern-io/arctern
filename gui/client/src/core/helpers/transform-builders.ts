import {
  Project,
  Aggregate,
  Filter,
  Bin,
  Sort,
  SortOrder,
  Limit,
  Expression,
  AliasExpression,
  AggregateFunctionExpression,
} from '../types';

export const project = (expr: string | {expr: string | Expression; as: string}): Project => ({
  type: 'project',
  expr: typeof expr === 'string' ? expr : expr.expr,
  as: typeof expr === 'string' ? undefined : expr.as,
});

export const getAggs = (agg: any) => {
  if (Array.isArray(agg)) {
    return {
      fields: agg.map(a => a.field),
      ops: agg.map(a => a.type),
      // $FlowFixMe
      as: agg.map(a => a.as),
    };
  } else {
    return {
      fields: [agg.field],
      ops: [agg.type],
      as: [agg.as || ''],
    };
  }
};

export const getGroupBy = (groupby: any) => {
  if (Array.isArray(groupby)) {
    return groupby.map(group => {
      if (typeof group === 'object') {
        return {
          type: 'project',
          expr: group.expr,
          as: group.as,
        };
      } else {
        return group;
      }
    });
  } else if (typeof groupby === 'object') {
    return {
      type: 'project',
      expr: groupby.expr,
      as: groupby.as,
    };
  } else {
    return groupby;
  }
};

export const aggregate = (
  groupby: AliasExpression | Array<AliasExpression> | string,
  agg: AggregateFunctionExpression | Array<AggregateFunctionExpression>
): Aggregate => {
  const aggs = getAggs(agg);
  const group = getGroupBy(groupby);
  return {
    type: 'aggregate',
    fields: aggs.fields,
    ops: aggs.ops,
    as: aggs.as,
    groupby: group,
  };
};

export const filter = (expr: string | Expression, id: string = ''): Filter => ({
  type: 'filter',
  expr,
});

export const filterRange = (
  field: string,
  range: Array<number | string>,
  id: string = ''
): Filter => ({
  type: 'filter',
  expr: {
    type: 'between',
    field,
    left: range[0],
    right: range[1],
  },
});

/**
 * Creates an Filter transform that uses an in expression
 * @memberof Transform
 */
export const filterIn = (field: string, set: Array<string | number>, id: string = ''): Filter => ({
  type: 'filter',
  expr: {
    type: 'in',
    expr: field,
    set,
  },
});

export const bin = (alias: string, field: string, extent: Array<number>, maxbins: number): Bin => ({
  type: 'bin',
  field,
  extent,
  maxbins,
  as: alias,
});

export const limit = (limit: number, offset?: number): Limit => ({
  type: 'limit',
  limit,
  offset,
});

export const sort = (field: string | Array<string>, order: SortOrder | Array<SortOrder>): Sort => ({
  type: 'sort',
  field: typeof field === 'string' ? [field] : field,
  order: typeof order === 'string' ? [order] : order,
});

export const top = (field: string, limit: number, offset?: number): [Sort, Limit] => [
  {
    type: 'sort',
    field: [field],
    order: ['descending'],
  },
  {
    type: 'limit',
    limit: limit,
    offset,
  },
];

export const bottom = (field: string, limit: number, offset?: number): [Sort, Limit] => [
  {
    type: 'sort',
    field: [field],
    order: ['ascending'],
  },
  {
    type: 'limit',
    limit: limit,
    offset,
  },
];
