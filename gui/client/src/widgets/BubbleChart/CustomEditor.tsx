import React, {FC} from 'react';
import {measureGetter} from '../../utils/WidgetHelpers';

const CustomEditor: FC = ({
  classes,
  MeasuresFormat,
  config,
  ColorPalette,
  setConfig,
  DimensionFormat,
  dataMeta,
  data,
}: any) => {
  const colorMeasure = measureGetter(config, 'color');

  return (
    <div className={classes.root}>
      <div className={classes.source}>
        <ColorPalette
          dataMeta={dataMeta}
          data={data}
          config={config}
          setConfig={setConfig}
          colorTypes={colorMeasure ? ['gradient'] : ['solid']}
        />
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
