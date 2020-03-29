import React, {FC} from 'react';

const CustomEditor: FC<any> = ({
  classes,
  MeasuresFormat,
  config,
  DimensionFormat,
  Sort,
  setConfig,
}) => {
  return (
    <div className={classes.root}>
      <div className={classes.source}>
        <Sort config={config} setConfig={setConfig} />
      </div>
      <div className={classes.source}>
        <MeasuresFormat config={config} setConfig={setConfig} />
      </div>
      <div className={classes.source}>
        <DimensionFormat config={config} setConfig={setConfig} />
      </div>
    </div>
  );
};

export default CustomEditor;
