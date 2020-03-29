import Parser from '.';
import {SQL, Transform} from '../types';

export default function parseHaving(sql: SQL, transform: Transform) {
  if (transform.type !== 'having') {
    return sql;
  }

  sql.having = sql.having || [];
  const expr = Parser.parseExpression(transform.expr);
  sql.having.push(expr);

  return sql;
}
