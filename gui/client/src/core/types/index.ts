import {InfiniNode} from '../infini';
export type NotExpression = {
  type: 'not';
  expr: string | BooleanExpression;
};

export type AndExpression = {
  type: 'and';
  left: string | BooleanExpression;
  right: string | BooleanExpression;
};
export type OrExpression = {
  type: 'or';
  left: string | BooleanExpression;
  right: string | BooleanExpression;
};
export type ComparisonOperatorExpression = {
  type: '=' | '<>' | '<' | '>' | '<=' | '>=';
  left: string | number;
  right: string | number;
};
export type BetweenExpression = {
  type: 'between' | 'not between';
  field: string;
  left: number | string;
  right: number | string;
};

export type NullExpression = {
  type: 'is null' | 'is not null';
  field: string;
};

export type PatternMatchingExpression = {
  type: 'like' | 'not like' | 'ilike';
  left: string;
  right: string;
};

export type InExpression = {
  type: 'in' | 'not in';
  expr: string;
  set: string | Array<string | number>;
};

export type CoalesceExpression = {
  type: 'coalesce';
  values: Array<string | null>;
};
export type CaseExpression = {
  type: 'case';
  cond: Array<[BooleanExpression | string, string]>;
  else: string;
};
export type CastExpresssion = {
  type: 'cast';
  expr: string;
  as: string;
};
export type StatisticalValueFunction = {
  type: 'stddev' | 'stddev_pop' | 'stddev_samp' | 'var_pop' | 'var_samp';
  x: string;
};
export type StatisticalPairFunction = {
  type: 'corr' | 'covar_pop' | 'covar_samp';
  x: string;
  y: string;
};
export type MaxExpression = {
  type: 'max';
  field: string;
  as?: string;
};

export type MinExpression = {
  type: 'min';
  field: string;
  as?: string;
};

// todo for the types like: map unique, now just string
export type UniqueExpression = {
  type: string;
  field?: string;
  as?: string;
  x?: string;
};

export type ProjectExpression = {
  type: 'project';
  field: string;
  as?: string;
};

export type SumExpression = {
  type: 'sum';
  field: string;
  as?: string;
};

export type AvgExpression = {
  type: 'average';
  field: string;
  as?: string;
};

export type CountExpression = {
  type: 'count';
  distinct: boolean;
  approx: boolean;
  field: string;
  as?: string;
};

export type DateTruncExpression = {
  type: 'date_trunc';
  unit: DateTruncUnits;
  field: string;
};
export type ExtractExpression = {
  type: 'extract';
  unit: ExtractUnits;
  field: string;
};
export type AliasExpression = {
  expr: string | Expression;
  as: string;
};

export type GisMappingExpression = {
  type: 'gis_mapping';
  domainEnd: number;
  field: string;
  domainStart: number;
  range: number;
};

export type GisTransScaleExpression = {
  type: 'gis_trans';
  domain: [number, number];
  field: string;
  width?: number;
  height?: number;
  range: number;
};

export type GisInCircleExpression = {
  type: 'in_circle';
  fromlat: string;
  distance: number;
  tolon: string;
  tolat: string;
};

export type TimeFunctionExpression = DateTruncExpression | ExtractExpression;
export type AggregateFunctionExpression =
  | MaxExpression
  | MinExpression
  | UniqueExpression
  | SumExpression
  | AvgExpression
  | CountExpression;
export type StatisticalFunctionExpression = StatisticalValueFunction | StatisticalPairFunction;
export type ConditionalExpression = CaseExpression | CoalesceExpression;
export type LogicalExpression = NotExpression | AndExpression | OrExpression;
export type ComparisonExpression =
  | ComparisonOperatorExpression
  | BetweenExpression
  | NullExpression;
export type BooleanExpression =
  | ComparisonExpression
  | LogicalExpression
  | ConditionalExpression
  | InExpression;
export type Expression =
  | LogicalExpression
  | ComparisonExpression
  | InExpression
  | ConditionalExpression
  | CastExpresssion
  | StatisticalFunctionExpression
  | AggregateFunctionExpression
  | PatternMatchingExpression
  | TimeFunctionExpression
  | ProjectExpression
  | GisMappingExpression
  | GisTransScaleExpression
  | GisInCircleExpression;

export type JoinRelation = 'join' | 'join.inner' | 'join.left' | 'join.right';
export type SortOrder = 'ascending' | 'descending' | 'asc' | 'desc';
export type ExtractUnits = 'year' | 'quarter' | 'month' | 'dom' | 'dow' | 'hour' | 'minute';
export type DateTruncUnits = 'decade' | 'year' | 'quarter' | 'month' | 'week' | 'day' | 'hour';
export type Aggregation = 'average' | 'count' | 'min' | 'max' | 'sum' | 'unique';
export type Condition =
  | 'between'
  | 'not between'
  | 'null'
  | 'not null'
  | 'equals'
  | 'not equals'
  | 'greater than or equals'
  | 'less than or equals'
  | 'equals'
  | 'not equals';

export type Bin = {
  type: 'bin';
  field: string;
  extent: Array<number>;
  maxbins: number;
  as: string;
};
export type With = {
  type: 'with';
  data: InfiniNode<Transform>;
  as?: string;
};
export type Limit = {
  type: 'limit';
  limit: number;
  offset?: number;
};
export type Sort = {
  type: 'sort';
  field: Array<string>;
  order?: Array<SortOrder>;
};
export type Filter = {
  type: 'filter';
  expr: any;
};

export type Having = {
  type: 'having';
  expr: string | Expression;
};
export type Project = {
  type: 'project';
  expr: string | Expression;
  as?: string;
};
export type Join = {
  type: JoinRelation;
  on?: Filter | Array<Filter>;
  as?: string;
};
export type Source = {
  type: 'source';
  source: string | Array<Join | Scan | InfiniNode<Transform>>;
};
export type Scan = {
  type: 'scan';
  table: string;
};
export type Crossfilter = {
  type: 'crossfilter';
  signal: string;
  filter: {[key: string]: Filter};
};
export type Aggregate = {
  type: 'aggregate';
  fields: Array<string>;
  ops: Array<Aggregation>;
  as: Array<string>;
  groupby: Array<string | Project | Bin> | string | Project | Bin;
};
export type ResolveFilter = {
  type: 'resolvefilter';
  filter: {signal: string};
  ignore?: Array<string | number> | string | number;
};

export type Transform =
  | Aggregate
  | Bin
  | Sort
  | Limit
  | Filter
  | Project
  | Crossfilter
  | With
  | Having
  | ResolveFilter
  | Source;

export type SQL = {
  select?: Array<string>;
  from?: string;
  where?: Array<string>;
  groupby?: Array<string>;
  having?: Array<string>;
  orderby?: Array<string>;
  limit?: number;
  offset?: number;
  unresolved?: {
    [key: string]: ResolveFilter;
  };
  with?: Array<any>;
};
