import Parser from '.';
import {SQL, Transform} from '../types';

function aggregateField(aggFn: string, field: string, as: string): string {
  if (aggFn === null) {
    return field;
  }
  const expression = Parser.parseExpression({
    type: aggFn,
    field,
  });
  return `${expression} ${as ? `AS ${as}` : ''}`;
}

function _parseGroupBy(sql: SQL, groupby: any) {
  if (typeof groupby === 'string') {
    sql.select = sql.select || [];
    sql.groupby = sql.groupby || [];
    sql.select.push(groupby);
    sql.groupby.push(groupby);
    return sql;
  }
  switch (groupby.type) {
    case 'bin':
      sql = Parser.parseTransform(sql, groupby);
      sql.groupby = sql.groupby || [];
      sql.groupby.push(groupby.as);
      break;
    case 'project':
      sql.select = sql.select || [];
      sql.select.push(
        Parser.parseExpression(groupby.expr) + (groupby.as ? ' AS ' + groupby.as : '')
      );
      if (groupby.as) {
        sql.groupby = sql.groupby || [];
        sql.groupby.push(groupby.as);
      }
      break;
    default:
      break;
  }
  return sql;
}

export default function parseAggregate(sql: SQL, transform: Transform) {
  if (transform.type === 'aggregate' && Array.isArray(transform.fields)) {
    if (Array.isArray(transform.groupby)) {
      transform.groupby.forEach((group: any) => {
        sql = _parseGroupBy(sql, group);
      });
    } else {
      sql = _parseGroupBy(sql, transform.groupby);
    }
    transform.fields.forEach((field: string, index: number) => {
      const as = transform.as[index];
      sql.select = sql.select || [];
      sql.select.push(aggregateField(transform.ops[index], field, as));
    });
  }

  return sql;
}
