import {SQL, Transform} from '../types';
import {reduceToString} from './reducer';

const parseJoin = (type: any) => {
  switch (type) {
    case 'join.left':
      return 'LEFT JOIN';
    case 'join.right':
      return 'RIGHT JOIN';
    case 'join.inner':
      return 'INNER JOIN';
    case 'join':
    default:
      return 'JOIN';
  }
};
const isScan = (transform: any) => transform.type === 'scan';
const isJoin = (transform: any) =>
  transform.type === 'join' ||
  transform.type === 'join.inner' ||
  transform.type === 'join.left' ||
  transform.type === 'join.right';

export default function parseSource(sql: SQL, transforms: Transform) {
  if (transforms.type !== 'source') {
    return sql;
  }
  sql.from = sql.from || '';
  if (typeof transforms.source === 'string') {
    sql.from = transforms.source;
    return sql;
  }
  sql.from = transforms.source
    .reduce((acc: any, transform: any) => {
      if (isScan(transform) && typeof transform.table === 'string') {
        return acc.concat(`${transform.table}${transform.as ? ` AS ${transform.as}` : ''}`);
      } else if (isJoin(transform)) {
        const right = acc.pop();
        const left = acc.pop();
        const joinString = `${left} ${parseJoin(transform.type)} ${right}`;
        return acc.concat(`${joinString}${transform.as ? ` AS ${transform.as}` : ''}`);
      } else if (transform.type === 'data' || transform.type === 'root') {
        const subquery = reduceToString(transform);
        return acc.concat(`(${subquery})`);
      } else {
        return acc;
      }
    }, [])
    .join();

  return sql;
}
