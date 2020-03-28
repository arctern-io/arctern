import React, {FC} from 'react';
import {measureGetter} from '../../utils/WidgetHelpers';
import {getColType} from '../../utils/ColTypes';

const CustomEditor: FC = ({
  classes,
  ColorPalette,
  config,
  setConfig,
  Limit,
  VisualDataMapping,
  nls,
}: any) => {
  const colorMeasure = measureGetter(config, 'color');
  const colType = colorMeasure && getColType(colorMeasure.type);
  const useColorPalette = !colorMeasure || (colorMeasure && colType === 'number');
  const useVisualDataMapping = colorMeasure && colType === 'text';

  return (
    <div className={classes.root}>
      <div className={classes.source}>
        <Limit
          min={1}
          max={50000}
          step={1}
          config={config}
          setConfig={setConfig}
          title={nls.label_of_points_size}
        />
      </div>
      {useColorPalette && (
        <div className={classes.source}>
          <ColorPalette
            config={config}
            setConfig={setConfig}
            colorTypes={[colorMeasure ? 'gradient' : 'solid']}
          />
        </div>
      )}
      {useVisualDataMapping && (
        <div className={classes.source}>
          <VisualDataMapping config={config} setConfig={setConfig} />
        </div>
      )}
    </div>
  );
};

export default CustomEditor;
