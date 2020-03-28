import Parser from '.';
import {SQL, Transform} from '../types';

export default function parseProject(sql: SQL, transform: Transform) {
  if (transform.type !== 'project') {
    return sql;
  }
  sql.select = sql.select || [];
  sql.select.push(
    Parser.parseExpression(transform.expr) + (transform.as ? ' AS ' + transform.as : '')
  );
  return sql;
}
