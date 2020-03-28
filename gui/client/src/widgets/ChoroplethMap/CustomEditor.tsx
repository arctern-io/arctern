import React, {FC} from 'react';

const CustomEditor: FC = ({
  classes,
  MeasuresFormat,
  config,
  ColorPalette,
  setConfig,
  MapTheme,
  data,
  dataMeta,
}: any) => {
  return (
    <div className={classes.root}>
      <div className={classes.source}>
        <MapTheme config={config} setConfig={setConfig} />
      </div>
      <div className={classes.source}>
        <ColorPalette
          dataMeta={dataMeta}
          data={data}
          config={config}
          setConfig={setConfig}
          colorTypes={['gradient']}
        />
      </div>
      <div className={classes.source}>
        <MeasuresFormat config={config} setConfig={setConfig} />
      </div>
    </div>
  );
};

export default CustomEditor;
