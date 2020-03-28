import parseFilter from './parseFilter';
import {SQL, Transform} from '../types';

const parseCrossfilter = (sql: SQL, transform: Transform) => {
  if (transform.type === 'crossfilter') {
    if (typeof sql.unresolved === 'object') {
      if (sql.unresolved.hasOwnProperty(transform.signal)) {
        Object.keys(transform.filter).forEach(key => {
          const filter = transform.filter[key];
          if (sql.unresolved) {
            const {ignore} = sql.unresolved[transform.signal];
            if (Array.isArray(ignore) ? ignore.indexOf(key) === -1 : key !== ignore) {
              sql = parseFilter(sql, filter);
            }
          }
        });
      }
    }
  }
  return sql;
};

export default parseCrossfilter;
