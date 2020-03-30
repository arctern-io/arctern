import {SQL, Transform} from '../types';

export default function parseLimit(sql: SQL, transform: Transform) {
  if (transform.type !== 'limit') {
    return sql;
  }
  sql.limit = sql.limit || 0;
  sql.offset = sql.offset || 0;

  sql.limit += transform.limit;
  sql.offset += transform.offset || sql.offset;
  return sql;
}
