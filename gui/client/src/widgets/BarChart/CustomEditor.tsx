import React, {FC} from 'react';
import {measureGetter} from '../../utils/WidgetHelpers';

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
  dataMeta,
  data,
}: any) => {
  const colorMeasure = measureGetter(config, 'color');
  const colorTypes = colorMeasure ? ['gradient'] : ['solid', 'ordinal'];
  return (
    <div className={classes.root}>
      <div className={classes.source}>
        <Sort config={config} setConfig={setConfig} />
      </div>
      <div className={classes.source}>
        <Limit
          attr={'limit'}
          initValue={10}
          min={2}
          max={100}
          config={config}
          title={nls.label_groups}
          setConfig={setConfig}
        />
      </div>
      <div className={classes.source}>
        <ColorPalette
          config={config}
          setConfig={setConfig}
          colorTypes={colorTypes}
          dataMeta={dataMeta}
          data={data}
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
