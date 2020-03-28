import {WIDGET} from '../utils/Consts';
import {cloneObj} from '../utils/Helpers';
import {Layout, WidgetConfig} from '../types';

export const fullLayoutWidth = 30;
export const fullLayoutHeight = 30;
export const defaultLayoutScales = [1 / 8, 1 / 4, 1 / 2, 1];
export const DEFAULT_WIDGET_WRAPPER_WIDTH = 800;
export const DEFAULT_WIDGET_WRAPPER_HEIGHT = 800;
export const breakPoints = {lg: 1200, md: 996, sm: 768, xs: 480, xxs: 0};

const MapWidget: string[] = [WIDGET.POINTMAP, WIDGET.GEOHEATMAP, WIDGET.CHOROPLETHMAP];

const _9avgSetLayout = (configs: WidgetConfig[]) => {
  return configs.map((config: WidgetConfig, index: number) => {
    const {layout} = config;
    layout.w = fullLayoutWidth / 3;
    layout.h = fullLayoutHeight / 3;
    layout.x = ((index % 3) * fullLayoutWidth) / 3;
    layout.y = (Math.floor(index / 3) * fullLayoutHeight) / 3;
    config.layout = layout;
    return config;
  });
};
const _4211SetLayout = (configs: WidgetConfig[]) => {
  return configs.map((config: WidgetConfig, index: number) => {
    const {layout} = config;
    switch (index) {
      case 0:
        layout.w = fullLayoutWidth / 2;
        layout.h = fullLayoutHeight;
        layout.x = 0;
        layout.y = 0;
        break;
      case 1:
        layout.w = fullLayoutWidth / 2;
        layout.h = fullLayoutHeight / 2;
        layout.x = fullLayoutWidth / 2;
        layout.y = 0;
        break;
      case 2:
        layout.w = fullLayoutWidth / 4;
        layout.h = fullLayoutHeight / 2;
        layout.x = fullLayoutWidth / 2;
        layout.y = fullLayoutHeight / 2;
        break;
      case 3:
        layout.w = fullLayoutWidth / 4;
        layout.h = fullLayoutHeight / 2;
        layout.x = (fullLayoutWidth * 3) / 4;
        layout.y = fullLayoutHeight / 2;
        break;
      default:
        let _index = index - 1;
        layout.x = ((_index % 3) * fullLayoutWidth) / 3;
        layout.y = fullLayoutHeight / 2 + (Math.floor(_index / 3) * fullLayoutHeight) / 3;
        layout.w = fullLayoutWidth / 3;
        layout.h = fullLayoutHeight / 3;
        break;
    }
    config.layout = layout;
    return config;
  });
};
const _1124SetLayout = (configs: WidgetConfig[]) => {
  return configs.map((config: WidgetConfig, index: number) => {
    const {layout} = config;
    switch (index) {
      case 0:
        layout.w = fullLayoutWidth / 4;
        layout.h = fullLayoutHeight / 2;
        layout.x = 0;
        layout.y = 0;
        break;
      case 1:
        layout.w = fullLayoutWidth / 4;
        layout.h = fullLayoutHeight / 2;
        layout.x = fullLayoutWidth / 4;
        layout.y = 0;
        break;
      case 2:
        layout.h = fullLayoutHeight / 2;
        layout.w = fullLayoutWidth / 2;
        layout.x = 0;
        layout.y = fullLayoutHeight / 2;
        break;
      case 3:
        layout.h = fullLayoutHeight;
        layout.w = fullLayoutWidth / 2;
        layout.x = fullLayoutWidth / 2;
        layout.y = 0;
        break;
      default:
        let _index = index - 1;
        layout.x = ((_index % 3) * fullLayoutWidth) / 3;
        layout.y = fullLayoutHeight / 2 + (Math.floor(_index / 3) * fullLayoutHeight) / 3;
        layout.w = fullLayoutWidth / 3;
        layout.h = fullLayoutHeight / 3;
        break;
    }
    config.layout = layout;
    return config;
  });
};
const _timelineTop = (configs: WidgetConfig[]) => {
  return configs.map((config: WidgetConfig, index: number) => {
    const {layout} = config;
    switch (index) {
      case 0:
        layout.w = fullLayoutWidth;
        layout.h = fullLayoutHeight / 6;
        layout.x = 0;
        layout.y = 0;
        break;
      case 1:
        layout.w = fullLayoutWidth / 2;
        layout.h = (fullLayoutHeight * 5) / 6;
        layout.x = 0;
        layout.y = fullLayoutHeight / 6;
        break;
      case 2:
        layout.w = fullLayoutWidth / 4;
        layout.h = fullLayoutHeight / 3;
        layout.x = fullLayoutWidth / 2;
        layout.y = fullLayoutHeight / 6;
        break;
      case 3:
        layout.h = fullLayoutHeight / 2;
        layout.w = fullLayoutWidth / 4;
        layout.x = fullLayoutWidth / 2;
        layout.y = fullLayoutHeight / 2;
        break;
      case 4:
        layout.h = (fullLayoutHeight * 5) / 6;
        layout.w = fullLayoutWidth / 4;
        layout.x = (fullLayoutWidth * 3) / 4;
        layout.y = fullLayoutHeight / 6;
        break;
      default:
        let _index = index - 5;
        layout.x = ((_index % 3) * fullLayoutWidth) / 3;
        layout.y = fullLayoutHeight + (Math.floor(_index / 3) * fullLayoutHeight) / 3;
        layout.w = fullLayoutWidth / 3;
        layout.h = fullLayoutHeight / 3;
        break;
    }
    config.layout = layout;
    return config;
  });
};
const _mapdDashboardSetLayout = (configs: WidgetConfig[]) => {
  return configs.map((config: WidgetConfig, index: number) => {
    const {layout} = config;
    switch (index) {
      case 0:
        layout.w = (fullLayoutWidth * 2) / 5;
        layout.h = fullLayoutHeight;
        layout.x = 0;
        layout.y = 0;
        break;
      case 1:
        layout.w = (fullLayoutWidth * 2) / 5;
        layout.h = (fullLayoutHeight * 4) / 12;
        layout.x = (fullLayoutWidth * 2) / 5;
        layout.y = 0;
        break;
      case 2:
        layout.w = fullLayoutWidth / 5;
        layout.h = fullLayoutHeight;
        layout.x = (fullLayoutWidth * 4) / 5;
        layout.y = 0;
        break;
      case 3:
        layout.w = fullLayoutWidth / 5;
        layout.h = (fullLayoutHeight * 5) / 12;
        layout.x = (fullLayoutWidth * 2) / 5;
        layout.y = (fullLayoutHeight * 4) / 12;
        break;
      case 4:
        layout.w = fullLayoutWidth / 5;
        layout.h = (fullLayoutHeight * 5) / 12;
        layout.x = (fullLayoutWidth * 3) / 5;
        layout.y = (fullLayoutHeight * 4) / 12;
        break;
      case 5:
        layout.w = fullLayoutWidth / 5;
        layout.h = (fullLayoutHeight * 3) / 12;
        layout.x = (fullLayoutWidth * 2) / 5;
        layout.y = (fullLayoutHeight * 9) / 12;
        break;
      case 6:
        layout.w = fullLayoutWidth / 5;
        layout.h = (fullLayoutHeight * 3) / 12;
        layout.x = (fullLayoutWidth * 3) / 5;
        layout.y = (fullLayoutHeight * 9) / 12;
        break;
      default:
        let _index = index - 1;
        layout.x = ((_index % 3) * fullLayoutWidth) / 3;
        layout.y = fullLayoutHeight + (Math.floor(_index / 3) * fullLayoutHeight) / 3;
        layout.w = fullLayoutWidth / 3;
        layout.h = fullLayoutHeight / 3;
        break;
    }
    return config;
  });
};
const LayoutMap = new Map([
  ['_9avg', {order: [MapWidget, WIDGET.LINECHART], setLayout: _9avgSetLayout}],
  ['_4211', {order: [MapWidget, WIDGET.LINECHART], setLayout: _4211SetLayout}],
  ['_1124', {order: ['', '', WIDGET.LINECHART, MapWidget], setLayout: _1124SetLayout}],
  ['_timelineTop', {order: [WIDGET.LINECHART, MapWidget], setLayout: _timelineTop}],
  [
    '_mapdDashboard',
    {
      order: [
        MapWidget,
        WIDGET.LINECHART,
        WIDGET.HEATCHART,
        WIDGET.BARCHART,
        WIDGET.PIECHART,
        WIDGET.NUMBERCHART,
        WIDGET.NUMBERCHART,
      ],
      setLayout: _mapdDashboardSetLayout,
    },
  ],
]);

const _sortConfigs = (configs: WidgetConfig[], orderArr: Array<string | string[]>) => {
  const cloneConfigs = cloneObj(configs);
  let res: Array<WidgetConfig | undefined> = [];
  orderArr.forEach((target: string | string[]) => {
    const index = cloneConfigs.findIndex((config: WidgetConfig) => {
      return Array.isArray(target)
        ? (target as string[]).some((_type: string) => _type === config.type)
        : target === config.type;
    });
    if (index > -1) {
      res.push(cloneConfigs[index]);
      cloneConfigs.splice(index, 1);
      return;
    }
    res.push(undefined);
  });
  res.forEach((item: WidgetConfig | undefined, index: number) => {
    if (!item && cloneConfigs.length > 0) {
      res[index] = cloneConfigs[0];
      cloneConfigs.shift();
    }
  });
  let finalRes = res.filter((item: WidgetConfig | undefined) => item !== undefined);
  return finalRes.concat(cloneConfigs);
};
const _sortConfigsWithLayoutType = (configs: WidgetConfig[], layoutType: string) => {
  return configs.length < 4 || layoutType === '_9avg'
    ? configs
    : _sortConfigs(configs, LayoutMap.get(layoutType)!.order);
};

export const applyUsedLayout = (configs: WidgetConfig[], layoutType: string) => {
  const orderedConfigs = cloneObj(_sortConfigsWithLayoutType(configs, layoutType));
  const setLayout = LayoutMap.get(layoutType)!.setLayout;
  return setLayout(orderedConfigs);
};

const _isValidLayout = (xPos: number, yPos: number, existLayouts: Layout[]) => {
  const [widthUnit, heightUnit] = [fullLayoutWidth / 3, fullLayoutHeight / 3];
  const centerPoint = [(xPos * 2 + widthUnit) / 2, (2 * yPos + heightUnit) / 2];
  const isInCurrLayout = existLayouts.some((layout: Layout) => {
    const {x, y, w, h} = layout;
    const isXPosIn = x <= centerPoint[0] && centerPoint[0] <= x + w;
    const isYPosIn = y <= centerPoint[1] && centerPoint[1] <= y + h;
    return isXPosIn && isYPosIn;
  });
  return !isInCurrLayout;
};

export const getInitLayout = (existLayouts: Layout[]) => {
  const [width, height] = [fullLayoutWidth, fullLayoutHeight];
  const [widthUnit, heightUnit] = [width / 3, height / 3];
  const newLayout = {
    w: widthUnit,
    h: heightUnit,
    static: false,
  };
  if (existLayouts.length === 0) {
    return {
      ...newLayout,
      x: 0,
      y: 0,
      minW: 3,
      minH: 1,
    };
  }
  let xPos: number = 0;
  let yPos: number = 0;
  let isValid: boolean = false;

  while (!isValid) {
    if (xPos + widthUnit >= width) {
      xPos = 0;
      yPos = yPos + heightUnit;
    } else {
      xPos = xPos + widthUnit;
    }
    isValid = _isValidLayout(xPos, yPos, existLayouts);
  }
  return {
    ...newLayout,
    x: xPos,
    y: yPos,
    minW: 3,
    minH: 1,
  };
};
