import {reducer} from './reducer';

function escapeQuotes(string: string) {
  if (typeof string === 'string') {
    return string.replace(/'/gi, "''");
  } else {
    return string;
  }
}

export default function parseExpression(expression: any): string {
  if (typeof expression === 'string') {
    return expression;
  }
  if (expression === null) {
    return 'NULL';
  }
  switch (expression.type) {
    case '=':
    case '<>':
    case '<':
    case '>':
    case '<=':
    case '>=':
      return `${expression.left} ${expression.type} ${
        typeof expression.right === 'string' ? `'${expression.right}'` : expression.right
      }`;
    case 'between':
    case 'not between':
      return `${expression.field} ${expression.type.toUpperCase()} ${
        typeof expression.left === 'string' ? `'${expression.left}'` : expression.left
      } AND ${typeof expression.right === 'string' ? `'${expression.right}'` : expression.right}`;
    case 'is null':
    case 'is not null':
      return `${expression.field} ${expression.type.toUpperCase()}`;
    case 'ilike':
    case 'like':
    case 'not like':
      return `${expression.left} ${expression.type.toUpperCase()} '%${expression.right}%'`;
    case 'coalesce':
      return `COALESCE(${expression.values
        .map((field: string) => parseExpression(field))
        .join(', ')})`;
    case 'in':
    case 'not in':
      if (Array.isArray(expression.set)) {
        return (
          expression.expr +
          ' ' +
          expression.type.toUpperCase() +
          ' (' +
          expression.set
            .map((field: number | string) =>
              typeof field === 'number' ? field : `'${escapeQuotes(field)}'`
            )
            .join(', ') +
          ')'
        );
      } else if (
        typeof expression.set === 'object' &&
        (expression.set.type === 'data' || expression.set.type === 'root')
      ) {
        return (
          expression.expr +
          ' ' +
          expression.type.toUpperCase() +
          ' (' +
          reducer(expression.set) +
          ')'
        );
      } else {
        return expression;
      }
    case 'not':
      return `NOT(${parseExpression(expression.expr)})`;
    case 'and':
    case 'or':
      return `(${parseExpression(
        expression.left
      )} ${expression.type.toUpperCase()} ${parseExpression(expression.right)})`;
    case 'case':
      const elseCase = expression.else === null ? 'NULL' : `'${expression.else}'`;
      return (
        'CASE WHEN ' +
        expression.cond
          .map((cond: any) => parseExpression(cond[0]) + ' THEN ' + cond[1])
          .join(' ') +
        (typeof expression.else !== 'undefined' ? ` ELSE ${elseCase}` : '') +
        ' END'
      );
    case 'date_trunc':
      return `date_trunc('${expression.unit}', ${expression.field})`;
    case 'extract':
      return `extract(${expression.unit} from ${expression.field})`;
    case 'root':
      return `(${reducer(expression)})`;
    case 'count':
      if (expression.distinct && expression.approx) {
        return `approx_count_distinct(${expression.field})`;
      } else if (expression.distinct) {
        return `count(distinct ${expression.field})`;
      } else {
        return `count(${expression.field})`;
      }
    case 'unique':
      return `count(distinct ${expression.field})`;
    case 'stddev':
    case 'stddev_pop':
    case 'stddev_samp':
    case 'var_pop':
    case 'var_samp':
      return `${expression.type}(${parseExpression(expression.expr)})`;
    case 'corr':
    case 'covar_pop':
    case 'covar_samp':
    case 'regr_avgx':
    case 'regr_avgy':
    case 'regr_count':
    case 'regr_intercept':
    case 'regr_r2':
    case 'regr_slope':
    case 'regr_sxx':
    case 'regr_sxy':
    case 'regr_syy':
      return `${expression.type}(${expression.y}, ${expression.x})`;
    case 'min':
    case 'max':
    case 'sum':
    case 'sample':
    case 'bool_and':
    case 'bool_or':
    case 'bit_and':
    case 'bit_or':
    case 'every':
    case 'avg':
      return `${expression.type}(${expression.field})`;
    case 'average':
      return 'avg(' + expression.field + ')';
    case 'polygon':
      return `is_in_polygon(${expression.x}, ${expression.y}, ARRAY[${expression.px}], ARRAY[${expression.py}])`;
    case 'gis_mapping_lon':
      return `gis_discrete_trans_scale_long_epsg_4326_900913 (${expression.domainStart}::float, ${
        expression.domainEnd
      }::float, ${expression.field}, ${Math.floor(expression.range)})`;
    case 'gis_mapping_lat':
      return `gis_discrete_trans_scale_lat_epsg_4326_900913 (${expression.domainStart}::float, ${
        expression.domainEnd
      }::float, ${expression.field}, ${Math.floor(expression.range)})`;
    case 'gis_discrete_trans_scale_w':
      return `gis_discrete_trans_scale(${expression.domain[0]}, ${
        expression.domain[1]
      }, 0, ${expression.width - 1}, ${expression.field}::float)`;
    case 'gis_discrete_trans_scale_h':
      return `gis_discrete_trans_scale(${expression.domain[0]}, ${
        expression.domain[1]
      }, 0, ${expression.height - 1}, ${expression.field}::float)`;
    case 'circle':
      return `is_in_circle(${expression.fromlon}, ${expression.fromlat}, ${expression.distance}, ${expression.tolon}, ${expression.tolat})`;
    case 'project':
      return `${expression.field}`;
    default:
      return expression;
  }
}
