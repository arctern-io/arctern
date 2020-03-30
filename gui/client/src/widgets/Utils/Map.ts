import {measureGetter, dimensionGetter} from '../../utils/WidgetHelpers';
import {MapChartConfig} from '../common/MapChart.type';

// Map related consts
export const DEFAULT_MAP_POINT_SIZE = 3;
export const DEFAULT_MAX_MAP_POINT_SIZE = 30;
export const DEFAULT_MAX_POINTS_NUM = 100000000;
export const DEFAULT_ZOOM = 4;
export const TRANSPARENT_PNG = `data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVQYV2NgYAAAAAMAAWgmWQ0AAAAASUVORK5CYII=
`;
export const NYC_CENTER: any = {
  lng: -73.94196940893625,
  lat: 40.70056427006182,
};
export const CHINA_CENTER: any = {
  lat: 34.5715158066769,
  lng: 106.9622862233764,
};

// A LngLatBounds object, an array of LngLatLike objects in [sw, ne] order, or an array of numbers in [west, south, east, north] order.
export const US_BOUNDS: any = [
  [-66.94, 49.38],
  [-124.39, 25.82],
];
export const CHINA_BOUNDS: any = [
  [73.499857, 18.060852],
  [134.77281, 53.560711],
];

export const NYC_ZOOM: number = 3.5867090422218464;
export const defaultMapCenterGetter = (language: string) => {
  return language === 'zh-CN' ? CHINA_CENTER : NYC_CENTER;
};

export const DEFAULT_RULER = {min: 0, max: 1000};

// Mapbox styles
export const MapThemes: any = [
  {
    label: 'Dark',
    value: 'mapbox://styles/mapbox/dark-v9',
  },
  {
    label: 'Light',
    value: 'mapbox://styles/mapbox/light-v9',
  },

  {
    label: 'Satellite',
    value: 'mapbox://styles/mapbox/satellite-v9',
  },
  {
    label: 'Streets',
    value: 'mapbox://styles/mapbox/streets-v10',
  },
  {
    label: 'Outdoors',
    value: 'mapbox://styles/mapbox/outdoors-v11',
  },

  {
    label: 'Guidance-Night',
    value: 'mapbox://styles/mapbox/navigation-guidance-night-v4',
  },
  {
    label: 'Guidance-Day',
    value: 'mapbox://styles/mapbox/navigation-guidance-day-v4',
  },
];

export const DefaultMapTheme = MapThemes[0];

export enum KEY {
  LONGTITUDE = 'lon',
  LATITUDE = 'lat',
  COLOR = 'color',
}

export const checkIsDraw = (filter: any) => {
  return ['st_distance', 'st_within'].some((t: string) => t === filter.expr.type);
};

// Map related helpers
export const mapboxCoordinatesGetter = (bounds: any) => {
  let northEast = [bounds._ne.lng, bounds._ne.lat];
  let southWest = [bounds._sw.lng, bounds._sw.lat];
  return [
    [southWest[0], northEast[1]],
    [northEast[0], northEast[1]],
    [northEast[0], southWest[1]],
    [southWest[0], southWest[1]],
  ];
};

// used to create a center point
// and its position data is from incoming data
export const markerPosGetter = (config: MapChartConfig, data: any, center: any = {}) => {
  center.lat = data[measureGetter(config, KEY.LATITUDE)!.value];
  center.lng = data[measureGetter(config, KEY.LONGTITUDE)!.value];

  return center;
};

// MapChart Single Point SQL Getter
export const shapeFileGetter = (config: any) => {
  const {zoom} = config;
  if (zoom < 8) {
    return 'district';
  }
  if (zoom >= 8 && zoom <= 13) {
    return 'block';
  }
  if (zoom > 13) {
    return 'building';
  }
  return '';
};

export const onMapLoaded = (config: MapChartConfig, getMapBound: Function) => {
  if (!config.bounds) {
    const lon = (measureGetter(config, 'lon') || dimensionGetter(config, 'lon'))!;
    const lat = (measureGetter(config, 'lat') || dimensionGetter(config, 'lat'))!;
    return getMapBound(lon.value, lat.value, config.source);
  }

  return Promise.resolve(-1);
};
