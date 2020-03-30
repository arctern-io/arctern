import React, {FC} from 'react';

const CustomEditor: FC<any> = ({
  classes,
  MeasuresFormat,
  config,
  ColorPalette,
  setConfig,
}: any) => {
  return (
    <div className={classes.root}>
      <div className={classes.source}>
        <MeasuresFormat config={config} setConfig={setConfig} />
      </div>
      <ColorPalette config={config} setConfig={setConfig} colorTypes={['solid']} />
    </div>
  );
};

export default CustomEditor;
