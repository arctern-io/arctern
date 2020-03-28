import {max, min, format, scaleLinear} from 'd3';
import {dateFormat} from './Formatters';
import {SqlParser as Parser} from 'infinivis-core';
import {WidgetConfig, Data, Dimension, Measure, Filters} from '../types';
import {OUT_OUT_CHART, COLUMN_TYPE, WIDGET} from './Consts';
import {d3TimeTranslation} from './Time';
import {cloneObj, isValidValue, formatSource} from './Helpers';
import {isDateCol, isNumCol} from './ColTypes';
import {autoNumDimensionFormat} from './Formatters';
import {TIME_BIN_INPUT_OPTIONS} from './Time';
import {getDefaultTitle} from './EditorHelper';
// sql expression parser
export const parseExpression = Parser.parseExpression;

// get single column data
const _dataGetter = (
  dataArray: Data,
  xKey: string,
  yKey: string,
  value: string,
  isTime: boolean,
  isByColorDimension: boolean
): Data => {
  let resArray: Data = [];
  isByColorDimension
    ? dataArray.forEach((d: any) => {
        if (d.color === yKey) {
          resArray.push({
            x: isTime ? new Date(d[xKey]) : d[xKey],
            y: d[value],
            key: yKey,
            as: yKey,
          });
        }
      })
    : (resArray = dataArray.map((d: any) => ({
        x: isTime ? new Date(d[xKey]) : d[xKey],
        y: d[yKey],
        key: yKey,
        as: yKey,
      })));

  return resArray;
};

// get xDomain only used in LineChart now
export const xDomainGetter = (
  xDimension: Dimension,
  xDomainFilterExpr: any = {},
  getMax: boolean = false
): any[] => {
  const normal: any = typeNormalizeObjGetter(dimensionTypeGetter(xDimension));
  const {left, right} = xDomainFilterExpr;
  const {min, max, extract} = xDimension;
  let [currMin, currMax] = xDimension.extent!;
  if (extract) {
    currMin = min!;
    currMax = max!;
  }
  return getMax ? [min, max].map(normal) : [left || currMin, right || currMax].map(normal);
};

// get yDomain
export const yDomainGetter = (seriesData: any, y: any): any[] => {
  let items = Object.keys(seriesData);
  let [minY, maxY] = y.domain();
  if (items.length === 0) return [minY, maxY];

  let newMinY: any, newMaxY: any;
  items.forEach((key: any) => {
    let dataArray = seriesData[key].map((d: any) => d.y * 1);
    let localMin = min(dataArray);
    let localMax = max(dataArray);
    if (newMinY === undefined) {
      newMinY = localMin || minY;
    }
    if (newMaxY === undefined) {
      newMaxY = localMax || maxY;
    }
    newMinY = min([newMinY, localMin]);
    newMaxY = max([newMaxY, localMax]);
  });
  // if start from min data. some chart will not show min data.
  // so yAxis need smaller than the min data if min data !== 0
  return [newMinY - 0 > 1 ? newMinY - 1 : newMinY, newMaxY].map((d: any) => d);
};

// get legend items
export const legendItemsGetter = (config: WidgetConfig) => {
  let items: any = [];
  const {type, colorItems = [], measures = []} = config;
  switch (type) {
    case WIDGET.LINECHART:
      const onlyMeasure = measures[0];
      colorItems.length > 0 &&
        colorItems.forEach((item: any) => {
          const {as, color, isRecords, label} = item;
          items.push({
            as: as, // need
            key: as, // need
            title: as,
            color,
            value: onlyMeasure.as,
            legendLabel: label,
            format: onlyMeasure.format || '',
            isRecords,
          });
        });
      break;
    default:
      config.measures.forEach((m: Measure) => {
        items.push({
          as: m.as,
          key: m.as,
          title: m.as,
          measure: m,
        });
      });
      break;
  }
  return items;
};

// in historgram we may get data less than maxbins , need to compelete the data
export const compeleteDataGetter = (
  data: any[],
  xDimension: Dimension,
  yMeasure: any,
  isTimeChart: boolean
) => {
  const {maxbins, type, as, min, max, extract} = xDimension;
  const {as: yAs} = yMeasure;
  const sortedData = data.sort((a: any, b: any) =>
    isTimeChart ? new Date(a[as]).getTime() - new Date(b[as]).getTime() : a[as] - b[as]
  );

  if (isNumCol(type) || extract) {
    let curIndex = extract ? Number(min) : 0;
    let newData: any = [];
    const maxIndex = extract ? max : Number(maxbins) - 1; // max data length

    sortedData.forEach((d: any) => {
      let index = parseInt(d[as]);
      while (curIndex < index) {
        newData.push({
          [as]: curIndex,
          [yAs]: NaN, // give you a sign to make height zero
        });
        curIndex++;
      }
      if (index <= curIndex) {
        newData.push(cloneObj(d));
        index === curIndex && curIndex++;
      }
    });

    while (curIndex <= (maxIndex as number)) {
      newData.push({
        [as]: curIndex,
        [yAs]: NaN,
      });
      curIndex++;
    }
    return newData;
  }
  return data;
};

export const parseDataToXDomain = (data: Data, xDimension: Dimension, selfFilter: any = {}) => {
  const {type, extent = [], as, maxbins} = xDimension;
  if (isNumCol(type)) {
    const [min, max] = extent;
    const {left, right} = selfFilter;
    const step = ((right || max) - (left || min)) / Number(maxbins);
    return data.map((d: any) => {
      const cloneD = cloneObj(d);
      const res = d[as] * step + (left || min);
      cloneD[as] = res;
      return cloneD;
    });
  }
  return data.sort((a: any, b: any) => {
    if (isDateCol(type)) {
      return new Date(a).getTime() - new Date(b).getTime();
    }
    return a - b;
  });
};

export const seriesDataGetter = (
  legendItems: any,
  data: Data,
  xKey: string,
  isTime: boolean,
  isByColorDimension: boolean = false
) => {
  let series: any = {};
  legendItems.forEach((item: any) => {
    let lineData: any[] = _dataGetter(data, xKey, item.as, item.value, isTime, isByColorDimension);
    series[item.key] = lineData;
  });
  return series;
};

// hover histogram data
export const histogramHoverGetter = (
  legendItems: any,
  seriesData: any,
  position: any,
  y: any,
  xDistance: Function,
  eq: Function,
  measureFormatterGetter: Function = defaultMeasureFormatterGetter
) => {
  let hoverData: any = {x: OUT_OUT_CHART, data: [], xV: null};
  if (position.x === OUT_OUT_CHART) {
    return hoverData;
  }
  // find correct item
  // which should be the closes item to the mouse position
  let minDistance: number = Infinity;
  let correctItem: any = null;
  let targetItems: any = [];
  legendItems.forEach((legend: any) => {
    let lineData = seriesData[legend.as];
    targetItems = lineData
      .filter((v: any) => v.positionX <= position.x)
      .sort((a: any, b: any) => new Date(a.x).getTime() - new Date(b.x).getTime());
    let target = targetItems[targetItems.length - 1];
    if (target) {
      let distance = Math.abs(xDistance(target.x, position.xV));

      if (!correctItem) {
        correctItem = target;
        minDistance = distance;
        return;
      }

      if (distance <= minDistance) {
        correctItem = target;
        minDistance = distance;
      }
    }
  });

  // get correct data
  legendItems.forEach((legend: any) => {
    let findSameX = [];
    if (correctItem) {
      let formatter = measureFormatterGetter(legend);

      findSameX = targetItems.filter(
        (v: any, i: number) => eq(v.x, correctItem.x) && i !== targetItems.length - 1
      );
      correctItem.formattedValue = formatter(correctItem.y);
      correctItem.color = legend.color;
      correctItem.yPos = y(correctItem.y * 1);

      hoverData.data = [correctItem, ...findSameX];
    } else {
      hoverData.data.push(null);
    }
  });

  if (correctItem && hoverData.data.length > 0) {
    hoverData.x = correctItem.positionX;
    hoverData.xV = correctItem.x;
  }

  if (hoverData.data.length === 0) {
    hoverData.x = OUT_OUT_CHART;
  }
  return hoverData;
};

// hover line series data
export const seriesHoverDataGetter = (
  legendItems: any,
  seriesData: any,
  position: any,
  x: any,
  y: any,
  localBisector: Function,
  xDistance: Function,
  eq: Function,
  measureFormatterGetter: Function = defaultMeasureFormatterGetter
) => {
  let hoverData: any = {x: OUT_OUT_CHART, data: [], xV: null};
  if (position.x === OUT_OUT_CHART) {
    return hoverData;
  }

  // find correct item
  // which should be the closes item to the mouse position
  let minDistance: number = Infinity;
  let correctItem: any = null;

  legendItems.forEach((legend: any) => {
    let lineData = seriesData[legend.as];
    let index = localBisector(lineData, position.xV);
    let target = lineData[index];

    if (target) {
      let distance = Math.abs(xDistance(target.x, position.xV));

      if (!correctItem) {
        correctItem = target;
        minDistance = distance;
        return;
      }

      if (distance <= minDistance) {
        correctItem = target;
        minDistance = distance;
      }
    }
  });

  // get correct data
  legendItems.forEach((legend: any) => {
    let lineData = seriesData[legend.as];
    let target = null;

    if (correctItem) {
      target = lineData.find((d: any) => {
        return eq(d.x, correctItem.x);
      });
    }

    if (target) {
      let formatter = measureFormatterGetter(legend);
      target.formattedValue = formatter(target.y);
      target.color = legend.color;
      target.yPos = y(target.y * 1);
      hoverData.data.push(target);
    } else {
      hoverData.data.push(null);
    }
  });

  if (correctItem && hoverData.data.length > 0) {
    hoverData.x = x(correctItem.x);
    hoverData.xV = correctItem.x;
  }

  if (hoverData.data.length === 0) {
    hoverData.x = OUT_OUT_CHART;
  }
  return hoverData;
};

// get time round d3
const timeRoundGetter = (binningResolution: any) => {
  return d3TimeTranslation[binningResolution];
};

export const lineChartBinDataGetter = (config: WidgetConfig, showBin: boolean) => {
  const dimensions = config.dimensions;

  return {
    showBin: showBin,
    bin: showBin ? dimensions[0].binningResolution : '',
  };
};

// time select data
export const timeSelectorDataGetter = (config: WidgetConfig, x: any, showTimeSelector: boolean) => {
  const range = config.filter.range;

  let timeSelectorData: any = {
    showTimeSelector: showTimeSelector,
    domain: x.domain(),
    selectTime: range ? [range.expr.left, range.expr.right] : x.domain(),
  };
  return timeSelectorData;
};

// brush data getter
export const brushDataGetter = (
  xDimension: Dimension,
  filter: Filters = {},
  isTimeBrush: boolean
) => {
  let brush: any = [];

  // get dfefault brush
  Object.keys(filter).some((f: string) => {
    let filterExpr: any = filter[f] && filter[f].expr;
    if (filterExpr && filterExpr.type === 'between' && f !== 'xDomain') {
      brush[0] = filterExpr.left;
      brush[1] = filterExpr.right;
      return true;
    } else {
      return false;
    }
  });

  // get timeBin
  const timeRound: any = isTimeBrush && timeRoundGetter(xDimension.binningResolution);
  const showBrush = brush && brush.length > 0;
  return {
    showBrush,
    isTimeBrush,
    timeRound,
    brush,
    hasBrush: brush.length > 0,
  };
};

// legend getter
export const legendDataGetter = (config: WidgetConfig, legendItems: any) => {
  let legendData: any = {
    showLegend: true,
    data: legendItems,
    source: config.source,
  };
  return legendData;
};

export const dimensionGetter = (config: WidgetConfig, as: string) => {
  return config.dimensions.find((d: Dimension) => d.as === as);
};

export const dimensionTypeGetter = (dimension: Dimension): COLUMN_TYPE => {
  if (isDateCol(dimension.type)) {
    return dimension.extract ? COLUMN_TYPE.NUMBER : COLUMN_TYPE.DATE;
  }

  if (isNumCol(dimension.type)) {
    return COLUMN_TYPE.NUMBER;
  }

  return COLUMN_TYPE.TEXT;
};

export const typeEqGetter = (type: COLUMN_TYPE): Function => {
  switch (type) {
    case COLUMN_TYPE.DATE:
      return (a: any, b: any) => {
        return new Date(a).getTime() === new Date(b).getTime();
      };
    case COLUMN_TYPE.NUMBER:
    case COLUMN_TYPE.TEXT:
    default:
      return (a: any, b: any) => {
        return a === b;
      };
  }
};

export const typeDistanceGetter = (type: COLUMN_TYPE): Function => {
  switch (type) {
    case COLUMN_TYPE.DATE:
      return (a: any, b: any): number => {
        return new Date(a).getTime() - new Date(b).getTime();
      };
    case COLUMN_TYPE.NUMBER:
    default:
      return (a: any, b: any): number => {
        return a - b;
      };
  }
};

export const typeNormalizeStringGetter = (type: COLUMN_TYPE): Function => {
  switch (type) {
    case COLUMN_TYPE.DATE:
      return (v: any) => dateFormat(new Date(v));
    case COLUMN_TYPE.NUMBER:
    case COLUMN_TYPE.TEXT:
    default:
      return (v: any) => {
        return v;
      };
  }
};

export const typeNormalizeObjGetter = (type: COLUMN_TYPE): Function => {
  switch (type) {
    case COLUMN_TYPE.DATE:
      return (v: any) => {
        return new Date(v);
      };
    case COLUMN_TYPE.NUMBER:
    case COLUMN_TYPE.TEXT:
    default:
      return (v: any) => {
        return v;
      };
  }
};

export const defaultMeasureFormatterGetter = (measure: any): Function => {
  let f: string = '.2f';

  if (measure.format === 'auto' || !measure.format) {
    return format(f);
  }
  return format(measure.format);
};

export const measureGetter = (config: WidgetConfig, as: string) => {
  return config.measures.find((m: Measure) => m.as === as);
};

export const yAxisFormatterGetter = (
  yMeasure: Measure,
  y: any,
  valueFormat: Function = format,
  autoFormatter: Function = autoNumDimensionFormat(y.domain())
): Function => {
  let formatter: Function = valueFormat;
  const formatString: string = yMeasure.format || 'auto';
  if (typeof formatString === 'string' && formatString !== 'auto') {
    formatter = valueFormat(formatString);
  } else if (formatString === 'auto') {
    formatter = autoFormatter;
  }
  return formatter;
};

export const throttle = (fn: Function, t: number = 1000) => {
  let timeout: any;
  let r: any = function(...args: any[]) {
    if (timeout) {
      clearTimeout(timeout);
    }
    timeout = setTimeout(function() {
      fn.apply(null, args);
    }, t);
  };

  r.cancel = () => {
    if (timeout) {
      clearTimeout(timeout);
    }
  };

  return r;
};

export const dimensionBetweenExprGetter = (dimension: any, range: any[] = []) => {
  const normal: any = typeNormalizeStringGetter(dimensionTypeGetter(dimension));
  range = range.map(normal);
  return {
    type: 'between',
    originField: dimension.value,
    field:
      dimension.timeBin && dimension.extract
        ? parseExpression({
            type: dimension.timeBin && dimension.extract ? 'extract' : 'date_trunc',
            field: dimension.value,
            unit: dimension.timeBin,
          })
        : dimension.value,
    left: range[0],
    right: range[1],
  };
};

export const dimensionEqExprGetter = (dimension: any, range: any[] = []) => {
  const normal: any = typeNormalizeStringGetter(dimensionTypeGetter(dimension));
  range = range.map(normal);
  return {
    type: '=',
    originField: dimension.value,
    left:
      dimension.timeBin && dimension.extract
        ? parseExpression({
            type: dimension.timeBin && dimension.extract ? 'extract' : 'date_trunc',
            field: dimension.value,
            unit: dimension.timeBin,
          })
        : dimension.value,
    right: range[0],
  };
};

export const rangeConfigGetter = (config: WidgetConfig) => {
  let copiedConfig = cloneObj(config);
  copiedConfig = {
    ...copiedConfig,
    id: config.linkId,
    type: WIDGET.RANGECHART,
    ignoreId: copiedConfig.id,
    ignore: ['range'],
    selfFilter: {
      ...copiedConfig.selfFilter,
      range: {}, // if not clear, rangechart will also change
    },
    filter: {}, // must set filter empty, or the filter on lineChart will effect itself
  };

  const xDimension = dimensionGetter(copiedConfig, 'x');
  if (!xDimension) {
    return copiedConfig;
  }
  return copiedConfig;
};

export const stackedBarYDomainGetter = (datas: any[], xDomain: any, stackType: string) => {
  let min: number = Infinity,
    max: number = -Infinity;

  // get min max from single data
  datas.forEach((data: any) => {
    min = Math.min(min, Number.parseFloat(data.y));
    max = Math.max(max, Number.parseFloat(data.y));
  });

  if (stackType === 'horizontal') {
    return {min, max};
  }
  // get min max from resGroupByXdomain
  const resGroupByXdomain: any = new Map();
  xDomain.forEach((val: any) => resGroupByXdomain.set(val, 0));
  datas.forEach((data: any) => {
    const val = resGroupByXdomain.get(data.x);
    resGroupByXdomain.set(data.x, val + Number.parseFloat(data.y));
  });
  resGroupByXdomain.forEach((value: any) => {
    min = Math.min(min, value);
    max = Math.max(max, value);
  });

  return {min, max};
};

export const stackedBarHoverDataGetter = (renderData: any[]) => {
  return renderData.map((data: any) => {
    const {x, y, color, setting = {}} = data;
    const {fill = ''} = setting;
    return {
      x,
      y,
      color,
      fill,
    };
  });
};

export function dimensionDataGetter(dimensions: any[], results: Data) {
  results = cloneObj(results);
  let numRows = results.length;
  let key = 'dimensionsData';
  for (let b = 0; b < dimensions.length; b++) {
    if (dimensions[b] === null) continue;

    const dimension = dimensions[b];
    const {isBinned, maxbins, min, max, extract, timeBin, as} = dimension;
    const normal: any = typeNormalizeObjGetter(dimensionTypeGetter(dimension));
    const binBounds: any = [min, max].map(normal);

    if (isBinned) {
      if (timeBin) {
        const timeBin = dimension.timeBin;

        if (extract) {
          for (let r = 0; r < numRows; ++r) {
            const result = results[r][as];
            if (result === null) {
              continue;
            }
            results[r][key] = results[r][key] || [];
            results[r][key].push({
              dimension: dimension,
              data: [result],
            });
          }
        } else {
          const currentBin = TIME_BIN_INPUT_OPTIONS.filter((b: any) => b.value === timeBin)[0];
          const intervalMs = currentBin.numSeconds * 1000;
          for (var r = 0; r < numRows; ++r) {
            const result = results[r][as];
            if (result === null) {
              continue;
            }

            const minValue = normal(result);
            const maxValue = new Date(minValue.getTime() + intervalMs - 1);

            results[r][key] = results[r][key] || [];
            results[r][key].push({
              dimension: dimension,
              data: [minValue, maxValue],
            });
          }
        }
      } else {
        let unitsPerBin = (binBounds[1] - binBounds[0]) / maxbins;
        for (let r = 0; r < numRows; ++r) {
          if (results[r][as] === null) {
            continue;
          }
          const min = results[r][as] * unitsPerBin + binBounds[0];
          const max = min + unitsPerBin;

          results[r][key] = results[r][key] || [];
          results[r][key].push({
            dimension: dimension,
            data: [min, max],
          });
        }
      }
    } else {
      for (let r = 0; r < numRows; ++r) {
        const result = results[r][as];
        if (result === null) {
          continue;
        }

        results[r][key] = results[r][key] || [];
        results[r][key].push({
          dimension: dimension,
          data: [result],
        });
      }
    }
  }
  return results;
}

export const getBinDateRange = (startDate: any, timeBin: string) => {
  let range: any[] = [];
  if (new Date(startDate).toString() === 'Invalid Date') {
    startDate = new Date().toISOString();
  }
  range[0] = new Date(startDate).toISOString();
  const targetOpt =
    TIME_BIN_INPUT_OPTIONS.find((item: any) => item.value === timeBin) || TIME_BIN_INPUT_OPTIONS[0];

  range[1] = new Date(new Date(startDate).getTime() + targetOpt.numSeconds * 1000).toISOString();
  return range;
};

export const getBinNumRange = (startNum: number, binsGroups: number, extent: number[]) => {
  let range: any[] = [],
    gap = (extent[1] - extent[0]) / binsGroups;
  range[0] = extent[0] + gap * startNum;
  range[1] = range[0] + gap;
  return range;
};

export const xyDomainGetter = (data: any[], atrri: string, range: any) => {
  let _min = min(data, d => Number.parseFloat(d[atrri])) || 0,
    _max = max(data, d => Number.parseFloat(d[atrri])) || 0,
    x = scaleLinear().range(range);
  if (_min >= 0) {
    x.domain([0, _max]);
  }
  if (_max <= 0) {
    x.domain([_max, 0]);
  }
  if (_min < 0 && _max > 0) {
    x.domain([_min, _max]);
  }
  x.range(range);
  return x;
};

export const numDomainGetter = (data: any[], atrri: string, range: any) => {
  let _min = min(data, d => Number.parseFloat(d[atrri])) || 0,
    _max = max(data, d => Number.parseFloat(d[atrri])) || 0,
    x = scaleLinear().range(range);
  x.domain([_min, _max]);
  x.range(range);
  return x;
};

export const getValidRulerBase = ({data, config}: any) => {
  if (data.length > 0 && isValidValue(data[0].color)) {
    let [rulerMin, rulerMax] = [
      min(data, (d: any) => Number.parseFloat(d.color)) || 0,
      max(data, (d: any) => Number.parseFloat(d.color)) || 0,
    ].map((item: any) => item.toFixed(2) * 1);

    if (rulerMin === rulerMax) {
      [rulerMin, rulerMax] = [rulerMin - 10, rulerMax + 10];
    }
    const {rulerBase = {}} = config;
    const currRulerBaseMin = rulerBase.min;
    const currRulerBaseMax = rulerBase.max;
    if (rulerMin !== currRulerBaseMin || rulerMax !== currRulerBaseMax) {
      return {min: rulerMin, max: rulerMax};
    }
    return null;
  }
  return null;
};

export const getExpression = (measure: Measure) => {
  const {value, expression, isCustom, isRecords, as} = measure;
  if (isCustom || !expression) {
    return `${value} as ${as}`;
  }
  if (isRecords) {
    return `COUNT(*) as ${as || 'countval'}`;
  }
  if (expression === 'unique') {
    return `COUUNT (DISTINCT ${value}) as ${as}`;
  }
  return `${expression}(${value}) as ${as}`;
};

export const popupContentGetter = (config: WidgetConfig, row: any) => {
  let content = `<ul>`;
  let staticPopupItems: any = [];
  config.measures.length > 0 &&
    config.measures.forEach((measure: Measure) => staticPopupItems.push(measure.label));
  (config.popupItems || [])
    .concat(staticPopupItems)
    .sort()
    .filter((str: string, index: number, arr: any) => str !== arr[index + 1])
    .forEach((item: string) => {
      const target = config.measures.find((measure: Measure) => measure.label === item);
      const value = target && target.isCustom ? row[target.as] : row[item];
      // const formatter = target ? formatterGetter(target) : undefined;
      content += `<li><span class="content-title"><strong>${item}:</strong></span><span>${value}</span></li>`;
    });

  return content + `</ul>`;
};

export const popupContentBuilder = (
  config: any,
  row: any,
  listClass: string = `content_detail_list`
) => {
  let content = `<ul class="${listClass}">`;
  Object.keys(row).forEach((key: any) => {
    // const target = config.measures.find((measure: any) => measure.value === key);
    content += `<li><span class="content-title"><strong>${key}</strong>:</span><span>${row[key]}</span></li>`;
  });

  return content + `</ul>`;
};

export const getWidgetTitle = (config: WidgetConfig, nls: any) => {
  const {title, source, measures = []} = config;
  if (title) {
    return title;
  }
  if (measures.length > 0) {
    const {expression, label} = getDefaultTitle(measures[0]);
    return `${nls[`label_widgetEditor_expression_${expression}`] || ''} ${label}`;
  }

  return formatSource(source);
};
