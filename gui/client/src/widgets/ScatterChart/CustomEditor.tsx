import React, {FC} from 'react';
import {measureGetter} from '../../utils/WidgetHelpers';
import {DEFAULT_MAP_POINT_SIZE, DEFAULT_MAX_MAP_POINT_SIZE} from '../Utils/Map';
import {getColType} from '../../utils/ColTypes';
const CustomEditor: FC = ({
  classes,
  MeasuresFormat,
  config,
  ColorPalette,
  VisualDataMapping,
  setConfig,
  MapTheme,
  Limit,
  PopUp,
  nls,
  options,
  dataMeta,
  data,
}: any) => {
  const colorMeasure = measureGetter(config, 'color');
  const colType = colorMeasure && getColType(colorMeasure.type);
  const useColorPalette = !colorMeasure || (colorMeasure && colType === 'number');
  const useVisualDataMapping = colorMeasure && colType === 'text';
  const popUpOpts: any = options.map((opt: any) => opt.colName);

  return (
    <div className={classes.root}>
      <div className={classes.source}>
        <MapTheme config={config} setConfig={setConfig} />
      </div>
      <div className={classes.source}>
        <Limit
          min={1000}
          max={1000000000}
          attr={'points'}
          config={config}
          setConfig={setConfig}
          title={nls.label_of_points_size}
        />
        <Limit
          min={DEFAULT_MAP_POINT_SIZE}
          max={DEFAULT_MAX_MAP_POINT_SIZE}
          attr={'pointSize'}
          title={nls.label_point_size}
          config={config}
          setConfig={setConfig}
        />
      </div>
      <div className={classes.source}>
        <PopUp config={config} setConfig={setConfig} options={popUpOpts} />
      </div>
      {useColorPalette && (
        <div className={classes.source}>
          <ColorPalette
            dataMeta={dataMeta}
            data={data}
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
      <div className={classes.source}>
        <MeasuresFormat config={config} setConfig={setConfig} />
      </div>
    </div>
  );
};

export default CustomEditor;
