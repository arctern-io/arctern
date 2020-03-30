import React, {FC} from 'react';
import {CONFIG} from '../../utils/Consts';
import HeatChart from '../HeatChart';
import {heatCellFilterHandler, heatMultiFilterHandler} from './filter';
import {HeatChartProps} from './type';

const HeatChartView: FC<HeatChartProps> = props => {
  const {config, setConfig} = props;
  const onCellClick = (cell: any) => {
    setConfig({
      type: CONFIG.UPDATE,
      payload: heatCellFilterHandler(config, cell),
    });
  };

  const filterMulti = (multiDimensionsData: any) => {
    setConfig({
      type: CONFIG.UPDATE,
      payload: heatMultiFilterHandler(config, multiDimensionsData),
    });
  };

  return (
    <HeatChart
      onCellClick={onCellClick}
      onRowClick={filterMulti}
      onColClick={filterMulti}
      {...props}
    />
  );
};

export default HeatChartView;
