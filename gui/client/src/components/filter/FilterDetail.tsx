import React from 'react';
import { SimpleSelector as Selector } from '../common/selectors';
import { makeStyles } from '@material-ui/core/styles';

const useStyles = makeStyles({
  root: {},
  type: {
    display: 'flex',
  },
  value: {}
})

const FilterTypes: string[] = ['between', 'equal',]
const FilterDetail = (props: any) => {
  const classes = useStyles({})
  const { filter } = props;

  return (
    <div className={classes.root}>
      {typeof filter === 'string'
        ? <p>{filter}</p>
        : (
          <>
            <div className={classes.type}>
              <Selector currOpt={{ value: filter.type }} options={FilterTypes} />
            </div>
            <div className={classes.value}>

            </div>
          </>
        )
      }
    </div>
  )
}

export default FilterDetail