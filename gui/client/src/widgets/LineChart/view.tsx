import React, {FC} from 'react';
import LineChart from '../LineChart';
import {LineChartProps} from './types';
import {CONFIG} from '../../utils/Consts';
import {lineRangeFilterHandler, rangeDomainHandler} from './filter';
import {rangeConfigGetter} from '../../utils/WidgetHelpers';

const LineChartNormal: FC<LineChartProps> = props => {
  const {setConfig, linkData, dataMeta, linkMeta, config} = props;
  // main chart range filter
  const onRangeChange = (config: any, range: any, isRange: boolean) => {
    const handler = isRange ? rangeDomainHandler : lineRangeFilterHandler;
    const newConfig = handler(config, range);
    if (newConfig) {
      setConfig({
        type: CONFIG.UPDATE,
        payload: newConfig,
      });
    }
  };

  const rangeConfig = rangeConfigGetter(config);
  return (
    <LineChart {...props} onRangeChange={onRangeChange} chartHeightRatio={2 / 3}>
      <LineChart
        config={rangeConfig}
        isRange={true}
        {...props}
        data={linkData!}
        dataMeta={linkMeta || dataMeta}
        onRangeChange={onRangeChange}
        chartHeightRatio={1 / 3}
      ></LineChart>
    </LineChart>
  );
};

export default LineChartNormal;
