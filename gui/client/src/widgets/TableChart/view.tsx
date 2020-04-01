import React, {FC} from 'react';
import {CONFIG} from '../../utils/Consts';
import {throttle} from '../../utils/WidgetHelpers';
import TableChart from '../TableChart';
import {TableChartProps} from './types';
import {handleFilter} from '../Utils/filters/common';
import {DEFAULT_SORT} from '../../components/settingComponents/Sort';

const onBottomReached = throttle((config: any, setConfig: Function) => {
  if (config.limit && config.filter >= 300) {
    return;
  }
  setConfig({type: CONFIG.ADD_LIMIT, payload: {limit: (config.limit || 0) + 25, id: config.id}});
}, 300);
const TableChartView: FC<TableChartProps> = props => {
  const {config, setConfig} = props;
  let {sort = DEFAULT_SORT, id} = config;

  const changeSort = (as: string) => {
    const payload =
      as === sort.name
        ? {
            ...sort,
            order: sort.order === 'ascending' ? 'descending' : 'ascending',
          }
        : {...sort, name: as};
    setConfig({
      type: CONFIG.ADD_SORT,
      payload: {...payload, id},
    });
  };

  const onColumnClick = (val: any, as: string) => {
    handleFilter({val, as, config, setConfig});
  };

  return (
    <TableChart
      {...props}
      onSortChange={changeSort}
      onColumnClick={onColumnClick}
      onBottomReached={() => {
        onBottomReached(config, setConfig);
      }}
    />
  );
};

export default TableChartView;
