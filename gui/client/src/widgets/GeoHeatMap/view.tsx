import React, {FC} from 'react';
import {useTheme} from '@material-ui/core/styles';
import {CONFIG} from '../../utils/Consts';
import {GeoHeatMapProps} from './types';
import MapboxGl from '../common/MapboxGl';
import {mapUpdateConfigHandler, drawUpdateConfigHandler} from '../Utils/filters/map';

const GeoHeatMapView: FC<GeoHeatMapProps> = props => {
  const theme = useTheme();
  const {config, setConfig} = props;

  // map update on bounds change
  const onMapUpdate = (map: any, container: any) => {
    const center = map.getCenter();
    const bounds = map.getBounds();
    const zoom = map.getZoom();
    const boundingClientRect = container.getBoundingClientRect();

    setConfig({
      type: CONFIG.UPDATE,
      payload: mapUpdateConfigHandler(config, {
        boundingClientRect,
        zoom,
        center,
        bounds,
      }),
    });
  };

  // on draw update
  const onDrawUpdate = (draws: any) => {
    setConfig({
      type: CONFIG.UPDATE,
      payload: drawUpdateConfigHandler(config, draws),
    });
  };

  return (
    <div
      className="z-map-chart"
      style={{
        backgroundColor: theme.palette.background.paper,
      }}
    >
      <MapboxGl
        {...props}
        onMapUpdate={onMapUpdate}
        onDrawUpdate={onDrawUpdate}
        draws={config.draws || []}
      />
    </div>
  );
};

export default GeoHeatMapView;
