import React, {FC} from 'react';

const CustomEditor: FC = ({
  classes,
  config,
  setConfig,
  MapTheme,
  MeasuresFormat,
  DimensionFormat,
}: any) => {
  return (
    <div className={classes.root}>
      <div className={classes.source}>
        <MapTheme config={config} setConfig={setConfig} />
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
