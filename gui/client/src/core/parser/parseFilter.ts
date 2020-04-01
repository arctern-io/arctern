import Parser from '.';
import {SQL, Transform} from '../types';

const isExpression = (expr: any) => {
  return Object.prototype.toString.call(expr) === '[object Object]';
};

export default function parseFilter(sql: SQL, transform: Transform) {
  if (transform.type !== 'filter') {
    return sql;
  }
  sql.where = sql.where || [];
  if (transform.type === 'filter') {
    sql.where.push(
      `(${isExpression(transform.expr) ? Parser.parseExpression(transform.expr) : transform.expr})`
    );
  }
  return sql;
}
