import {scaleOrdinal, interpolateRgbBasis} from 'd3';
import {isValidValue} from './Helpers';
import {WidgetConfig} from '../types';

const makeScale = (num: number, colorGroup: any) => {
  return colorGroup.slice(0, num);
};

const EchartRange = [
  '#37A2DA',
  '#32C5E9',
  '#67E0E3',
  '#9FE6B8',
  '#FFDB5C',
  '#ff9f7f',
  '#fb7293',
  '#E062AE',
  '#E690D1',
  '#e7bcf3',
  '#9d96f5',
  '#8378EA',
  '#96BFFF',
];
type Ruler = {
  min: number;
  max: number;
};
export const allColorRange = EchartRange;
export const DEFAULT_RULER = {min: 0, max: 100};
export const UNSELECTED_COLOR = '#aaa';

const _ordinal2 = makeScale(2, allColorRange);
const _ordinal3 = makeScale(3, allColorRange);
const _ordinal4 = makeScale(4, allColorRange);
const _ordinal8 = makeScale(8, allColorRange);

type ColorOption = {
  key: string;
  value: string | string[];
};
export const solidOpts = makeScale(8, allColorRange).map((color: string) => {
  return {key: color, value: color};
});

export const ordinalOpts = [
  {key: 'ordinal2', value: _ordinal2},
  {key: 'ordinal3', value: _ordinal3},
  {key: 'ordinal4', value: _ordinal4},
  {key: 'ordinal8', value: _ordinal8},
];

export const gradientOpts = [
  {
    key: 'blue_green_yellow',
    value: [
      'rgb(17, 95, 154)',
      'rgb(25, 132, 197)',
      'rgb(34, 167, 240)',
      'rgb(72, 181, 196)',
      'rgb(118, 198, 143)',
      'rgb(166, 215, 91)',
      'rgb(201, 229, 47)',
      'rgb(208, 238, 17)',
      'rgb(208, 244, 0)',
    ], // blue_green_yellow
  },
  {
    key: 'blue_to_red',
    value: [
      'rgb(0, 0, 255)',
      'rgb(28, 0, 226)',
      'rgb(56, 0, 198)',
      'rgb(85, 0, 170)',
      'rgb(113, 0, 141)',
      'rgb(141, 0, 113)',
      'rgb(170, 0, 85)',
      'rgb(198, 0, 56)',
      'rgb(226, 0, 28)',
    ],
  },
  {
    key: 'purple_to_yellow',
    value: [
      'rgb(128, 0, 128)',
      'rgb(142, 28, 113)',
      'rgb(156, 56, 99)',
      'rgb(170, 85, 85)',
      'rgb(184, 113, 71)',
      'rgb(198, 141, 56)',
      'rgb(212, 170, 42)',
      'rgb(226, 198, 28)',
      'rgb(240, 226, 14)',
    ],
  },
  {
    key: 'white_blue',
    value: [
      'rgb(226, 226, 226)',
      'rgb(197, 218, 229)',
      'rgb(162, 208, 232)',
      'rgb(126, 198, 236)',
      'rgb(90, 187, 239)',
      'rgb(62, 179, 240)',
      'rgb(34, 167, 240)',
      'rgb(25, 132, 197)',
      'rgb(17, 95, 154)',
    ], // white_blue
  },
  {
    key: 'blue_white_red',
    value: [
      'rgb(25, 132, 197)',
      'rgb(34, 167, 240)',
      'rgb(99, 191, 240)',
      'rgb(167, 213, 237)',
      'rgb(226, 226, 226)',
      'rgb(225, 166, 146)',
      'rgb(222, 110, 86)',
      'rgb(225, 75, 49)',
      'rgb(194, 55, 40)',
    ], // blue_white_red
  },
  {
    key: 'green_yellow_red',
    value: [
      'rgb(77, 144, 79)',
      'rgb(90, 166, 81)',
      'rgb(137, 188, 85)',
      'rgb(191, 211, 89)',
      'rgb(237, 225, 91)',
      'rgb(237, 179, 78)',
      'rgb(236, 124, 63)',
      'rgb(225, 75, 49)',
      'rgb(194, 55, 40)',
    ], // green_yellow_red
  },
];

export const allColorOpts = [...solidOpts, ...ordinalOpts, ...gradientOpts];

export const DEFAULT_COLOR = allColorRange[0];

export const color = scaleOrdinal(allColorRange);

const _getValidColorValue = (value: string | number, {min, max}: Ruler = DEFAULT_RULER) => {
  let _value = Number.parseFloat(value as string);
  if (_value < min) {
    _value = min;
  }
  if (_value > max) {
    _value = max;
  }
  return (_value - min) / (max - min);
};

const _getColorOpt = (key: string, colorOpts: ColorOption[]) => {
  return colorOpts.find((opt: ColorOption) => opt.key === key);
};
const _getColorType = (colorKey: string) => {
  if (_getColorOpt(colorKey, gradientOpts)) {
    return 'gradient';
  }
  if (_getColorOpt(colorKey, ordinalOpts)) {
    return 'ordinal';
  }
  if (_getColorOpt(colorKey, solidOpts)) {
    return 'solid';
  }
  return '';
};

export const isGradientType = (colorKey: string) => _getColorType(colorKey) === 'gradient';

const _isValidColorKey = (colorKey: string, colorTypes: string[]) => {
  const colorType = _getColorType(colorKey);
  return !!colorTypes.find((c: string) => c === colorType);
};
const _getDefaultColorKey = (colorType: string) => {
  switch (colorType) {
    case 'gradient':
      return gradientOpts[0].key;
    case 'ordinal':
      return ordinalOpts[0].key;
    case 'solid':
    default:
      return solidOpts[0].key;
  }
};
export const getValidColorKey = (colorKey: string, colorTypes: string[]) => {
  return _isValidColorKey(colorKey, colorTypes) ? colorKey : _getDefaultColorKey(colorTypes[0]);
};

const _genGradientColorGetter = (config: WidgetConfig, defaultColor: string = '#000'): Function => {
  const {ruler, colorKey = ''} = config;
  const colorValues = getColorValues(colorKey);
  const colorGetter = interpolateRgbBasis(colorValues);
  return (value: number | string | undefined | null) =>
    isValidValue(value)
      ? colorGetter(_getValidColorValue(value as string | number, ruler))
      : defaultColor;
};
const _genOrdinalColorGetter = (config: WidgetConfig) => {
  const {colorKey = ''} = config;
  const colorValues = getColorValues(colorKey);
  return scaleOrdinal(colorValues);
};
export const getColorValues = (colorKey: string) => {
  return (allColorOpts.find((opt: ColorOption) => opt.key === colorKey) || solidOpts[0]).value;
};
export const genColorGetter = (config: WidgetConfig): Function => {
  const {colorKey = DEFAULT_COLOR} = config;
  const colorType = _getColorType(colorKey);
  switch (colorType) {
    case 'gradient':
      return _genGradientColorGetter(config);
    case 'ordinal':
      return _genOrdinalColorGetter(config);
    case 'solid':
    default:
      return () => colorKey;
  }
};
