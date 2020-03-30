import React, {useState, useEffect, useRef, useContext, useMemo} from 'react';
import clsx from 'clsx';
import {Theme, makeStyles, useTheme} from '@material-ui/core/styles';
import Table from '@material-ui/core/Table';
import TableBody from '@material-ui/core/TableBody';
import TableCell from '@material-ui/core/TableCell';
import TableHead from '@material-ui/core/TableHead';
import TableSortLabel from '@material-ui/core/TableSortLabel';
import TableRow from '@material-ui/core/TableRow';
import Paper from '@material-ui/core/Paper';
import Spinner from '../../components/common/Spinner';
import {cloneObj} from '../../utils/Helpers';
import {isSelected} from '../Utils/filters/common';
import {getBinDateRange, getBinNumRange, measureGetter} from '../../utils/WidgetHelpers';
import {formatterGetter} from '../../utils/Formatters';
import {Dimension, Measure} from '../../types';
import {I18nContext} from '../../contexts/I18nContext';
import {TableChartProps} from './types';
import {DEFAULT_SORT} from '../../components/settingComponents/Sort';

const useStyles = makeStyles((theme: Theme) => ({
  root: {
    flexGrow: 1,
    display: 'flex',
    flexDirection: 'column',
    maxWidth: '100%',
    overflowY: 'auto',
    background: theme.palette.background.paper,
  },
  paper: {
    width: '100%',
    overflowX: 'auto',
    overflowY: 'auto',
    marginBottom: theme.spacing(2),
    backgroundColor: theme.palette.background.default,
  },
  table: {
    backgroundColor: theme.palette.background.default,
  },
  customPagination: {
    backgroundColor: theme.palette.background.default,
  },
  customCell: {
    cursor: 'pointer',
    borderBottomColor: theme.palette.grey[400],
    backgroundColor: theme.palette.background.default,
    fontFamily: 'NotoSansCJKsc-Bold,NotoSansCJKsc',
    fontWeight: `bold`,
  },
  head: {
    color: `rgba(176,176,185,1)`,
  },
  body: {
    color: `rgba(0,0,0,1)`,
  },
  selectedCell: {
    color: theme.palette.primary.main,
  },
}));

const TableChart = (props: TableChartProps) => {
  const {nls} = useContext(I18nContext);
  const theme = useTheme();
  const classes = useStyles(theme);
  const {
    isLoading,
    data,
    config,
    onColumnClick,
    onSortChange,
    onBottomReached,
    dataMeta,
    wrapperWidth,
    wrapperHeight,
  } = props;
  const cloneConfig = cloneObj(config);
  const {sort = DEFAULT_SORT} = cloneConfig;
  const noData = !data || data.length === 0;
  const [content, setContent] = useState<any>([]);
  const root = useRef<any>(null);
  const headers = useMemo(() => {
    const res: any[] = [];
    const {dimensions = [], measures = []} = config;
    dimensions.forEach((d: Dimension) => {
      const {label} = d;
      res.push({
        label,
        target: d,
      });
    });
    measures.forEach((m: Measure) => {
      let {label, isCustom, expression} = m;
      if (!isCustom) {
        label = `${label}(${nls[`label_widgetEditor_expression_${expression}`]})`;
      }
      res.push({
        label,
        target: m,
      });
    });
    return res;
    //eslint-disable-next-line react-hooks/exhaustive-deps
  }, [JSON.stringify([...config.dimensions, ...config.measures])]);

  const effectors = JSON.stringify([
    dataMeta,
    config.dimensions,
    config.measures,
    theme,
    wrapperWidth,
    wrapperHeight,
  ]);
  useEffect(() => {
    const renderDatas = data.map((_data: any) => {
      const renderData: any = {};
      headers.forEach((header: any) => {
        const {target} = header;
        const {timeBin = '', extract = false, maxbins, as, extent} = target as Dimension;
        renderData[as] = {};
        renderData[as].originData = _data[as];
        const formatter = formatterGetter(target);
        const isTimeBin = timeBin && !extract;
        const isNumBin = !!maxbins;
        if (isTimeBin) {
          renderData[as].value = getBinDateRange(_data[as], timeBin)
            .map((val: any) => formatter(val))
            .join(' ~ ');
        } else if (isNumBin) {
          renderData[as].value = getBinNumRange(_data[as], maxbins!, extent as number[])
            .map((val: any) => formatter(val))
            .join(' ~ ');
        } else {
          renderData[as].value = formatter(_data[as]);
        }
        renderData[as].isSelected = measureGetter(config, as)
          ? false
          : isSelected(_data[as], as, config);
      });
      return renderData;
    });
    const _content = (
      <Paper className={classes.paper} ref={root}>
        <Table className={classes.table} size="small">
          <TableHead>
            <TableRow hover={true}>
              {headers.map((header: any, index: number) => (
                <TableCell key={index} classes={{root: clsx(classes.customCell, classes.head)}}>
                  <TableSortLabel
                    active={sort.name === header.target.as}
                    direction={sort.order === 'descending' ? 'desc' : 'asc'}
                    onClick={() => {
                      onSortChange(header.target.as);
                    }}
                  >
                    {header.label}
                  </TableSortLabel>
                </TableCell>
              ))}
            </TableRow>
          </TableHead>
          <TableBody ref={root}>
            {renderDatas.map((renderData: any, index: number) => {
              return (
                <TableRow key={index} hover={true}>
                  {headers.map((header: any, index: number) => {
                    const {isSelected, value, originData} = renderData[header.target.as];
                    return (
                      <TableCell
                        classes={{
                          root: clsx(classes.customCell, {[classes.selectedCell]: isSelected}),
                        }}
                        key={index}
                        onClick={() => {
                          if (measureGetter(config, header.target.as)) {
                            return;
                          }
                          onColumnClick(originData, header.target.as);
                        }}
                      >
                        {value}
                      </TableCell>
                    );
                  })}
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      </Paper>
    );
    const _onBottomReached = (e: any) => {
      const isTouchBottom =
        root.current.scrollHeight - root.current.scrollTop - root.current.clientHeight <= 1;
      if (isTouchBottom) {
        onBottomReached && onBottomReached();
      }
      return false;
    };
    let container: any;
    if (root.current) {
      container = root.current;
      container.addEventListener('scroll', _onBottomReached);
    }
    setContent(_content);
    return () => {
      container && container.removeEventListener('scroll', _onBottomReached);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [effectors]);

  if (isLoading) {
    return <Spinner />;
  }

  if (noData && !isLoading) {
    return <></>;
  }

  return <div className={classes.root}>{content}</div>;
};

export default TableChart;
