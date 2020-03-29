import React, {FC} from 'react';

const CustomEditor: FC = ({
  classes,
  MeasuresFormat,
  config,
  ColorPalette,
  setConfig,
  DimensionFormat,
  data,
  dataMeta
}: any) => {
  return (
    <div className={classes.root}>
      <div className={classes.source}>
        <MeasuresFormat config={config} setConfig={setConfig} />
      </div>
      <div className={classes.source}>
        <DimensionFormat config={config} setConfig={setConfig} />
      </div>
      <div className={classes.source}>
        <ColorPalette
          config={config}
          setConfig={setConfig}
          colorTypes={['gradient']}
          dataMeta={dataMeta}
          data={data}
        />
      </div>
    </div>
  );
};

export default CustomEditor;
