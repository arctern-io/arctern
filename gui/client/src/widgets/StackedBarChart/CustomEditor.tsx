import React, {FC} from 'react';

const CustomEditor: FC<any> = ({
  classes,
  MeasuresFormat,
  config,
  Sort,
  StackType,
  setConfig,
  VisualDataMapping,
}: any) => {
  return (
    <div className={classes.root}>
      <div className={classes.source}>
        <Sort config={config} setConfig={setConfig} />
      </div>
      <div className={classes.source}>
        <StackType config={config} setConfig={setConfig} />
      </div>
      <div className={classes.source}>
        <VisualDataMapping config={config} setConfig={setConfig} />
      </div>
      <div className={classes.source}>
        <MeasuresFormat config={config} setConfig={setConfig} />
      </div>
    </div>
  );
};

export default CustomEditor;
