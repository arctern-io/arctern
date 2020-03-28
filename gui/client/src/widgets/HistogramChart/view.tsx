import React, {FC, useEffect} from 'react';
import HistogramChart from '../HistogramChart';
import {HistogramChartProps} from './types';
import {CONFIG} from '../../utils/Consts';
import {lineRangeFilterHandler} from '../LineChart/filter';

const HistogramChartView: FC<HistogramChartProps> = props => {
  const {
    setConfig,
    linkData,
    config,
    data,
    dataMeta,
    linkMeta,
    mode,
    setMode,
    dashboard,
    wrapperHeight,
    wrapperWidth,
  } = props; // main chart range filter
  const isShowRange = config.isShowRange;
  const onRangeChange = (config: any, range: any, isRangeChart: boolean) => {
    const newConfig = lineRangeFilterHandler(config, range, isRangeChart);
    if (newConfig) {
      setConfig({
        type: CONFIG.UPDATE,
        payload: newConfig,
      });
    }
  };

  // clear selfFilter if crossFilter is clear
  useEffect(() => {
    if (JSON.stringify(config.filter) === '{}' && config.selfFilter.range) {
      setConfig({type: CONFIG.DEL_SELF_FILTER, payload: {id: config.id, filterKeys: ['range']}});
    }
    //eslint-disable-next-line react-hooks/exhaustive-deps
  }, [config.filter]);

  return (
    <HistogramChart
      wrapperHeight={wrapperHeight}
      wrapperWidth={wrapperWidth}
      setConfig={setConfig}
      dashboard={dashboard}
      mode={mode}
      setMode={setMode}
      config={config}
      data={data}
      dataMeta={dataMeta}
      linkMeta={linkMeta}
      chartHeightRatio={2 / 3}
      showXLabel={!isShowRange}
      onRangeChange={onRangeChange}
    >
      {isShowRange && (
        <HistogramChart
          wrapperHeight={wrapperHeight}
          wrapperWidth={wrapperWidth}
          setConfig={setConfig}
          dashboard={dashboard}
          mode={mode}
          setMode={setMode}
          config={config}
          data={linkData!}
          dataMeta={dataMeta}
          linkMeta={linkMeta}
          chartHeightRatio={1 / 3}
          isRange={true}
          showXLabel={true}
          onRangeChange={onRangeChange}
        ></HistogramChart>
      )}
    </HistogramChart>
  );
};

export default HistogramChartView;
