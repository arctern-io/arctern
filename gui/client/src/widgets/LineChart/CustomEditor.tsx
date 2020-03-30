import React, {FC} from 'react';
import FormControlLabel from '@material-ui/core/FormControlLabel';
import Switch from '@material-ui/core/Switch';
import {id} from '../../utils/Helpers';

const CustomEditor: FC = ({
  classes,
  MeasuresFormat,
  config,
  StyleSwitch,
  setConfig,
  DimensionFormat,
  VisualDataMapping,
  nls,
}: any) => {
  const onRangeShow = () => {
    config.linkId = id();
    const isShowRange = !config.isShowRange;
    if (!isShowRange) {
      delete config.selfFilter.range;
    }
    setConfig({payload: {...config, isShowRange}});
  };

  return (
    <div className={classes.root}>
      <div className={classes.source}>
        <StyleSwitch config={config} setConfig={setConfig} />
      </div>
      <div className={classes.source}>
        <MeasuresFormat config={config} setConfig={setConfig} />
      </div>
      <div className={classes.source}>
        <DimensionFormat config={config} setConfig={setConfig} />
      </div>
      <div className={classes.source}>
        <FormControlLabel
          control={<Switch checked={config.isShowRange} onChange={onRangeShow} />}
          label={nls.label_show_range}
        />
      </div>
      <div className={classes.source}>
        <VisualDataMapping config={config} setConfig={setConfig} />
      </div>
    </div>
  );
};

export default CustomEditor;
