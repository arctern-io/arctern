import {SQL, Transform} from '../types';

export default function parseResolvefilter(sql: SQL, transform: Transform) {
  if (transform.type === 'resolvefilter') {
    sql.unresolved = sql.unresolved || {};
    sql.unresolved[transform.filter.signal] = transform;
  }
  return sql;
}
