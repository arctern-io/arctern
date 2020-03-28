import React, {FC, useState} from 'react';
import clsx from 'clsx';
import {makeStyles, useTheme} from '@material-ui/core/styles';
import UITable from '@material-ui/core/Table';
import TableHead from '@material-ui/core/TableHead';
import TableBody from '@material-ui/core/TableBody';
import TableRow from '@material-ui/core/TableRow';
import TableCell from '@material-ui/core/TableCell';
import TableSortLabel from '@material-ui/core/TableSortLabel';
import TablePagination from '@material-ui/core/TablePagination';
import {SORT_ORDER} from '../../utils/Consts';
import {genBasicStyle} from '../../utils/Theme';

const useStyles = makeStyles(theme => ({
  ...genBasicStyle(theme.palette.primary.main),
  link: {
    textDecoration: 'none',
  },
  customCell: {
    borderBottomColor: theme.palette.grey[400],
    backgroundColor: theme.palette.background.default,
    fontFamily: 'NotoSansCJKsc-Bold,NotoSansCJKsc',
    fontWeight: `bold`,
  },
  head: {
    color: `rgba(176,176,185,1)`,
  },
}));

export type TableCommonProps = {
  data: any[];
  length: number;
  def?: any[];
  isUsePagination?: boolean;
};

const Table: FC<TableCommonProps> = props => {
  const theme = useTheme();
  const classes = useStyles(theme);
  // get data, def, length from props
  const {data = [], def = [], length = 0, isUsePagination = false} = props;
  const [order, setOrder] = useState<SORT_ORDER>(SORT_ORDER.ASC);
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(5);
  const {field = ''} = def[0] || {};
  const [orderBy, setOrderBy] = useState(field as string);

  const defaultDef =
    data.length > 0
      ? Object.keys(data[0]).map((key: string) => {
          return {
            field: key,
            name: key,
            format: null,
            onClick: () => {},
          };
        })
      : [];

  const renderHeaders: any[] = (def.length === 0 ? defaultDef : def).map((header: any) => {
    const {field = '', name = '', sortable = false} = header;
    const sortDirection = order === field ? (order === SORT_ORDER.ASC ? 'asc' : 'desc') : false;
    const active = orderBy === field;
    const direction = order === SORT_ORDER.ASC ? 'asc' : 'desc';

    return {field, name, sortDirection, active, direction, sortable};
  });

  data.sort((row1, row2) => {
    return order === SORT_ORDER.ASC
      ? typeof row1[orderBy] === 'string'
        ? row1[orderBy].localeCompare(row2[orderBy])
        : -1
      : typeof row2[orderBy] === 'string'
      ? row2[orderBy].localeCompare(row1[orderBy])
      : -1;
  });

  const createSortHandler = (header: string) => {
    header === orderBy
      ? setOrder(order === SORT_ORDER.DESC ? SORT_ORDER.ASC : SORT_ORDER.DESC)
      : setOrderBy(header);
  };

  const handleChangePage = (event: any, newPage: number) => {
    setPage(newPage);
  };

  const handleChangeRowsPerPage = (event: any) => {
    setRowsPerPage(event.target.value * 1);
    setPage(0);
  };

  if (!data || data.length === 0) {
    return <div>no data</div>;
  }

  return (
    <div className="dashboards">
      <UITable stickyHeader size="small">
        <TableHead>
          <TableRow>
            {renderHeaders.map((header: any) => {
              const {field, name, sortDirection, active, direction, sortable} = header;
              return (
                <TableCell
                  classes={{
                    body: classes.customCell,
                    head: clsx(classes.customCell, classes.head),
                    stickyHeader: classes.customCell,
                  }}
                  key={field}
                  sortDirection={sortDirection}
                >
                  {sortable && (
                    <TableSortLabel
                      active={active}
                      direction={direction}
                      onClick={() => {
                        createSortHandler(field);
                      }}
                    >
                      {name.toUpperCase()}
                    </TableSortLabel>
                  )}
                  {!sortable && name.toUpperCase()}
                </TableCell>
              );
            })}
          </TableRow>
        </TableHead>
        <TableBody>
          {data.map((row: any, rowIndex: number) => (
            <TableRow key={rowIndex} hover={true}>
              {def.map((currDef: any, cellIndex: number) => {
                const v = currDef.field;
                const currOnClick = currDef ? currDef.onClick : null;
                const hasWidget = typeof currDef.widget === 'function';
                let content = currDef ? (currDef.format ? currDef.format(row[v]) : row[v]) : row[v];
                content = hasWidget ? currDef.widget(row, rowIndex, cellIndex) : content;

                return (
                  <TableCell
                    classes={{
                      root: currOnClick ? classes.hover : '',
                      body: classes.customCell,
                    }}
                    key={cellIndex}
                    onClick={() => {
                      if (!hasWidget) {
                        currOnClick && currOnClick(row, rowIndex, cellIndex);
                      }
                    }}
                  >
                    {content}
                  </TableCell>
                );
              })}
            </TableRow>
          ))}
        </TableBody>
      </UITable>
      {isUsePagination && (
        <TablePagination
          rowsPerPageOptions={[5, 10, 25]}
          component="div"
          count={length}
          rowsPerPage={rowsPerPage}
          page={page}
          backIconButtonProps={{
            'aria-label': 'Previous Page',
          }}
          nextIconButtonProps={{
            'aria-label': 'Next Page',
          }}
          onChangePage={handleChangePage}
          onChangeRowsPerPage={handleChangeRowsPerPage}
        />
      )}
    </div>
  );
};

export default Table;
