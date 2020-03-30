import React, { Fragment } from 'react';
import { makeStyles } from '@material-ui/core/styles';
import FilterCard from './FilterCard'
const _saveFilter = (key: string, parsedFilter: any, res: any) => {
  res[key] = res[key] || [];
  res[key].push(parsedFilter);
  return res
}
const _parseFilter = (filter: any) => {
  let res: any = {};
  Object.keys(filter).forEach((key: string) => {
    const _filter = filter[key];
    const { type, expr } = _filter;
    if (type !== 'filter') {
      return;
    }
    if (typeof expr === 'string') {
      res = { ...res }
    }
    switch (expr.type) {
      case 'between':
        res = _saveFilter(expr.originField, expr, res);
        break;
      case 'in':
      default:
        break;
    }
  })
  return res;
}

const useStyles = makeStyles({
  root: {
    position: 'absolute',
    width: '200px',
    minHeight: '400px',
    // background: 'red',
    zIndex: 1000000
  },
})
const FilterList = (props: any) => {
  const classes = useStyles({})
  const { config } = props;
  const result = _parseFilter(config.filter);
  console.log(result)
  return (
    <div className={classes.root}>
      {Object.keys(result).map((key: string) => {
        const filters = result[key];
        return (
          <Fragment key={key}>
            <FilterCard column={key} filters={filters} />
          </Fragment>
        )
      })}
    </div>
  )
}

export default FilterList;