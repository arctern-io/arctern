import {WidgetProps, BaseWidgetConfig, Dimension, Measure} from '../../types';

export type MapDimension = Dimension & {
  domainStart: number;
  domainEnd: number;
  field: string;
  range: number;
};

export type MapMeasure = Measure & {
  domainStart: number;
  domainEnd: number;
  field: string;
  range: number;
};
export type MapChartConfig = BaseWidgetConfig<MapDimension, MapMeasure> & {
  zoom: number;
  mapTheme: string;
  width: number;
  height: number;
  center: any;
  bounds: any;
  pointSize?: number;
  popupItems?: any;
  ruler?: any;
  filter: {
    [key: string]: any;
  };
  selfFilter: {
    xBounds: any;
    yBounds: any;
    [key: string]: any;
  };
};

export type MapChartProps = WidgetProps<MapChartConfig> & {
  draws?: any;
  onMapUpdate?: Function;
  onDrawUpdate?: Function;
  onMouseMove?: Function;
  onMouseOut?: Function;
  onMapClick?:Function;
  onMapLoaded?: Function;
  allowPopUp?: boolean;
  showRuler?: boolean;
};
