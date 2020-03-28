import React, {FC} from 'react';

const CustomEditor: FC<any> = ({config, setConfig, DimensionFormat, classes}) => {
  return (
    <div className={classes.root}>
      <div className={classes.source}>
        <DimensionFormat config={config} setConfig={setConfig} />
      </div>
    </div>
  );
};

export default CustomEditor;
