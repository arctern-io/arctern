import React, { Fragment } from 'react';
import FilterDetail from './FilterDetail'
import { makeStyles } from '@material-ui/core/styles';

const useStyles = makeStyles({
  root: {},
  title: {},
  content: {}
})
const FilterCard = (props: any) => {
  const classes = useStyles({})
  const { column, filters } = props;
  return (
    <div className={classes.root}>
      <div className={classes.title}>{column}</div>
      <div className={classes.content}>
        {filters.map((filter: any, index: number) => (
          <Fragment key={index}><FilterDetail filter={filter} /></Fragment>
        ))}
      </div>
    </div>
  )

}

export default FilterCard