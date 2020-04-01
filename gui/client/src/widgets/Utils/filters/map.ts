import {measureGetter, dimensionGetter} from '../../../utils/WidgetHelpers';
import {MapChartConfig} from '../../common/MapChart.type';
import {cloneObj} from '../../../utils/Helpers';
import {KEY, checkIsDraw} from '../Map';

export const mapUpdateConfigHandler = (
  config: MapChartConfig,
  {boundingClientRect, zoom, center, bounds}: any
) => {
  let copiedConfig = cloneObj(config);
  // console.log(bounds, zoom);
  const {width, height} = boundingClientRect;
  // console.log("add draw", bounds);
  copiedConfig.zoom = zoom;
  copiedConfig.center = center;
  copiedConfig.bounds = bounds;
  copiedConfig.width = width;
  copiedConfig.height = height;
  return copiedConfig;
};

export const drawUpdateConfigHandler = (config: MapChartConfig, draws: any) => {
  const copiedConfig = cloneObj(config);
  const {filter = {}} = copiedConfig;
  // get required lon, lat
  const lon = dimensionGetter(config, KEY.LONGTITUDE) || measureGetter(config, KEY.LONGTITUDE);
  const lat = dimensionGetter(config, KEY.LATITUDE) || measureGetter(config, KEY.LATITUDE);

  // clear all draws
  Object.keys(filter).forEach((f: any) => {
    if (checkIsDraw(filter[f])) {
      delete filter[f];
    }
  });

  draws.forEach((draw: any) => {
    if (draw.data.properties.isCircle) {
      filter[draw.id] = {
        type: 'filter',
        expr: {
          type: 'st_distance',
          fromlon: draw.data.properties.center[0],
          fromlat: draw.data.properties.center[1],
          tolon: lon!.value,
          tolat: lat!.value,
          distance: draw.data.properties.radiusInKm * 1000,
        },
      };
      return;
    }

    if (draw.type === 'Polygon') {
      if (draw.data.geometry.coordinates[0][0] === null) {
        return;
      }
      filter[draw.id] = {
        type: 'filter',
        isGeoJson: true,
        expr: {
          type: 'st_within',
          geoJson: draw.data,
          x: lon!.value,
          y: lat!.value,
          px: draw.data.geometry.coordinates[0].map((point: any) => point[0]),
          py: draw.data.geometry.coordinates[0].map((point: any) => point[1]),
        },
      };
    }
  });
  copiedConfig.filter = filter;
  copiedConfig.draws = draws;
  return copiedConfig;
};
