import React, {Fragment} from 'react';
import {useTheme} from '@material-ui/core/styles';
import {format} from 'd3';
import {makeStyles} from '@material-ui/core/styles';
import {customOptsStyle} from '../../utils/Theme';
import {formatSource} from '../../utils/Helpers';
import {Source, DataCache} from '../../types';

const useStyles = makeStyles(theme => ({
  ...customOptsStyle,
  container: {
    display: 'flex',
    justifyContent: 'center',
    margin: '5px',
  },
  wrapper: {
    fontFamily: `NotoSansCJKsc-Bold,NotoSansCJKsc`,
    fontWeight: `bold`,
    marginRight: theme.spacing(2),
    fontSize: `1rem`,
  },
}));

interface ISourceStats {
  data: DataCache;
  sources: Source[];
  totals: number[];
}

const SourceStats = (props: ISourceStats) => {
  const {data, sources, totals} = props;
  const classes = useStyles() as any;
  const theme = useTheme();
  // selected counts
  const selected: any = sources
    .map((s: string) => {
      let count = data[s];
      if (count) {
        let total: number = (totals as any)[s] || 0;
        let c = count[0] && count[0].countval;
        let label = (
          <span className={classes.wrapper}>
            <span style={{color: theme.palette.primary.main}}>{`${format(',.0f')(c || 0)}`}</span>
            <span>{` of ${format(',.0f')(total)} · ${formatSource(s)}`}</span>
          </span>
        );
        return {
          id: s,
          label,
        };
      }
      return false;
    })
    .filter((s: any) => s);
  return (
    <div className={classes.container}>
      {selected.map((s: any) => {
        return <Fragment key={s.id}>{s.label}</Fragment>;
      })}
    </div>
  );
};

export default SourceStats;
