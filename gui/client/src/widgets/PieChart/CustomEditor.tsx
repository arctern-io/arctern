import React, {FC} from 'react';

const CustomEditor: FC = ({
  classes,
  MeasuresFormat,
  config,
  ColorPalette,
  setConfig,
  Sort,
  Limit,
  DimensionFormat,
  nls,
}: any) => {
  const colorTypes = ['solid', 'ordinal'];

  return (
    <div className={classes.root}>
      <div className={classes.source}>
        <Sort config={config} setConfig={setConfig} />
      </div>
      <div className={classes.source}>
        <Limit
          attr={'limit'}
          min={1}
          max={100}
          config={config}
          title={nls.label_groups}
          setConfig={setConfig}
        />
      </div>
      <div className={classes.source}>
        <ColorPalette config={config} setConfig={setConfig} colorTypes={colorTypes} />
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
