import React from 'react';
import {SimpleSelector as Selector} from '../common/selectors';
import {cloneObj, formatSource} from '../../utils/Helpers';
import {CONFIG} from '../../utils/Consts';

const Source = (props: any) => {
  const {config, setConfig, options, onMouseOver} = props;
  const cloneConfig = cloneObj(config);
  const selectorOpts = options.map((o: any) => {
    return {
      label: formatSource(o),
      value: o,
    };
  });
  const onSourceChange = (val: string) => {
    if (val === cloneConfig.source) {
      return false;
    }
    // we need to delete map bound
    // so that if the source user select has different location
    // we can move our map to that bounds
    if (cloneConfig.isServerRender) {
      delete cloneConfig.bounds;
    }
    cloneConfig.dimensions.length = 0;
    cloneConfig.measures.length = 0;
    cloneConfig.source = val;
    cloneConfig.colorItems = [];
    setConfig({type: CONFIG.REPLACE_ALL, payload: cloneConfig});
  };

  return (
    <Selector
      currOpt={{label: formatSource(config.source), value: config.source}}
      options={selectorOpts}
      onOptionChange={onSourceChange}
      onMouseOver={onMouseOver}
      isShowCurrOpt={true}
    />
  );
};

export default Source;
