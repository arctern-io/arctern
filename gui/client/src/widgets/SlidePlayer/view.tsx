import React, {FC, useMemo} from 'react';
import SlidePlayer from './index';
import {Dimension} from '../../types';
import {getColType} from '../../utils/ColTypes';
import {lineRangeFilterHandler} from '../LineChart/filter';
import {CONFIG} from '../../utils/Consts';
import {SlidePlayerProps} from './types';

const SlidePlayerView: FC<SlidePlayerProps> = props => {
  const {config, setConfig, dataMeta, wrapperWidth, wrapperHeight} = props;
  const data = useMemo(() => {
    return config.dimensions.map((d: Dimension) => {
      const {value, as, type} = d;
      let options: any = [];
      switch (getColType(type)) {
        case 'text':
          options = d.options || [];
          break;
        case 'number':
          options = d.extent;
          break;
        case 'date':
          const {extract, min: currMin, max: currMax} = d;
          if (extract) {
            options = [1, currMax];
          } else {
            options = [new Date(currMin!), new Date(currMax!)];
          }
          break;
        default:
          break;
      }
      return {
        value,
        as,
        options,
      };
    });
  }, [config.dimensions]);

  const onRangeChange = (config: any, range: any) => {
    const newConfig = lineRangeFilterHandler(config, range);
    if (newConfig) {
      setConfig!({
        type: CONFIG.UPDATE,
        payload: newConfig,
      });
    }
  };
  return (
    <SlidePlayer
      wrapperWidth={wrapperWidth}
      wrapperHeight={wrapperHeight}
      config={config}
      dataMeta={dataMeta}
      onRangeChange={onRangeChange}
      data={data}
    />
  );
};

export default SlidePlayerView;
