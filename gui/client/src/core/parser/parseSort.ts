import {SQL, Transform} from '../types';

const ORDER_DIRECTIONS: any = {
  ascending: 'ASC',
  asc: 'ASC',
  descending: 'DESC',
  desc: 'DESC',
  undefined: 'ASC',
};

export default function parseSort(sql: SQL, transform: Transform) {
  if (transform.type !== 'sort') {
    return sql;
  }
  sql.orderby = sql.orderby || [];
  transform.field.forEach((field: any, index: number) => {
    sql.orderby.push(
      field + (Array.isArray(transform.order) ? ' ' + ORDER_DIRECTIONS[transform.order[index]] : '')
    );
  });
  return sql;
}
