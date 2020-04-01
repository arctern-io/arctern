import {MapChartConfig} from '../common/MapChart.type';
import {DEFAULT_RULER, DEFAULT_MAP_POINT_SIZE, shapeFileGetter} from './Map';
import {color, DEFAULT_COLOR, getValidColorKey} from '../../utils/Colors';
import {color as d3Color, RGBColor} from 'd3';

type colorItem = {
  label: string;
  color: string;
};

const labelColorVegaGen = (colorItem: colorItem) => {
  const _color = d3Color(colorItem.color) as RGBColor;

  return {
    label: colorItem.label,
    color_r: _color.r,
    color_g: _color.g,
    color_b: _color.b,
    color_a: _color.opacity,
  };
};

export const vegaGen = (config: MapChartConfig, colorTypes: string[]) => {
  const {colorKey = '', width, height, pointSize} = config;
  const validColorKey = getValidColorKey(colorKey, colorTypes);
  const _color = d3Color(validColorKey || DEFAULT_COLOR) as RGBColor;

  let vega: any = {
    width: Math.floor(width),
    height: Math.floor(height),
    data: [
      {name: 'render_type', values: ['circles_2d']},
      {
        name: 'colors',
        values: [
          {
            color_r: _color.r,
            color_g: _color.g,
            color_b: _color.b,
            color_a: _color.opacity,
          },
        ],
      },
      {
        name: 'radius',
        values: [pointSize || DEFAULT_MAP_POINT_SIZE],
      },
      {name: 'image_format', values: ['png']},
    ],
  };
  return JSON.stringify(vega);
};

export const vegaPointGen = (config: MapChartConfig) => {
  let items: any = config.colorItems || [];
  let colorsItems = items
    .map((c: any) => ({
      label: c.as,
      color: c.color || color(c.as),
    }))
    .map(labelColorVegaGen);

  let vega: any = {
    width: Math.floor(config.width),
    height: Math.floor(config.height),
    data: [
      {name: 'render_type', values: ['multi_color_circles_2d']},
      {
        name: 'colors',
        values: colorsItems,
      },
      {
        name: 'radius',
        values: colorsItems.map(() => config.pointSize || DEFAULT_MAP_POINT_SIZE),
      },
      {name: 'image_format', values: ['png']},
    ],
  };
  return JSON.stringify(vega);
};

export const vegaPointWeihtedGen = (config: MapChartConfig, colorTypes: string[]) => {
  const {ruler, colorKey = '', width, height, pointSize} = config;
  const validColorKey = getValidColorKey(colorKey, colorTypes);
  const {min, max} = ruler || DEFAULT_RULER;
  let vega: any = {
    width: Math.floor(width),
    height: Math.floor(height),
    data: [
      {name: 'render_type', values: ['weighted_color_circles_2d']},
      {
        name: 'color_style',
        values: [
          {
            style: validColorKey,
            ruler: [min, max],
          },
        ],
      },
      {
        name: 'radius',
        values: [pointSize || DEFAULT_MAP_POINT_SIZE],
      },
      {name: 'image_format', values: ['png']},
    ],
  };
  return JSON.stringify(vega);
};

export const vegaHeatMapGen = (config: MapChartConfig) => {
  const {zoom, width, height} = config;

  let vega: any = {
    width: Math.floor(width),
    height: Math.floor(height),
    data: [
      {name: 'render_type', values: ['heatmap_2d']},
      {
        name: 'map_scale_ratio',
        values: [zoom],
      },
      {name: 'image_format', values: ['png']},
    ],
  };
  return JSON.stringify(vega);
};

export const vegaChoroplethMapGen = (config: MapChartConfig) => {
  const {width, height, colorKey, selfFilter, ruler} = config;
  const geo_type_value = shapeFileGetter(config);
  const {xBounds, yBounds} = selfFilter;

  let vega: any = {
    width: Math.floor(width),
    height: Math.floor(height),
    data: [
      {name: 'render_type', values: ['building_weighted_2d']},
      {
        name: 'color_style',
        values: [{style: colorKey}, {ruler: [ruler.min, ruler.max]}],
      },
      {
        name: 'geo_type',
        values: geo_type_value,
      },
      {
        name: 'bound_box',
        values: [xBounds.expr.left, yBounds.expr.left, xBounds.expr.right, yBounds.expr.right],
      },
      {name: 'image_format', values: ['png']},
    ],
  };
  return JSON.stringify(vega);
};
