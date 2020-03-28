import {SQL, Transform} from '../types';

export default function parseBin(sql: SQL, transform: Transform) {
  if (transform.type !== 'bin') {
    return sql;
  }
  // numBins is used conditionally in our query building below.
  // first of all, if we're going to fall into the overflow bin AND we have
  // 0 bins, then we should land in bin 0. Otherwise, we should land in the last
  // bin.
  //
  // later, we calculate the binning magic number based on numBins - dividing either
  // by it or 1 if it doesn't exist, to prevent a divide by zero / infinity error.
  const {field, as, extent, maxbins} = transform;
  const numBins = extent[1] - extent[0];

  let caseSql = `CASE WHEN ${field} >= ${extent[1]} THEN ${
    numBins === 0 ? 0 : maxbins - 1
  } else cast((cast(${field} AS float) - ${extent[0]}) * ${maxbins / (numBins || 1)} AS int) end`;

  sql.select = sql.select || [];
  sql.where = sql.where || [];
  sql.having = sql.having || [];

  sql.select.push(`${caseSql} AS ${as}`);
  sql.where.push(`((${field} >= ${extent[0]} AND ${field} <= ${extent[1]}) OR (${field} IS NULL))`);
  sql.having.push(`((${caseSql}) >= 0 AND (${caseSql}) < ${maxbins} OR (${caseSql}) IS NULL)`);
  return sql;
}
