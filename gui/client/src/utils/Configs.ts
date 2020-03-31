import {
  InfiniCollection,
  InfiniNode,
  SqlParser,
  Helper,
  ResolveFilter,
  Crossfilter,
  Transform,
} from 'infinivis-core';
import {rangeConfigGetter} from './WidgetHelpers';
import {DEFAULT_COLOR} from '../utils/Colors';
import {DEFAULT_CHART, COLUMN_TYPE} from './Consts';
import {id, cloneObj} from './Helpers';
import {getInitLayout} from '../utils/Layout';
import {
  WidgetConfig,
  Layout,
  Query,
  QueryType,
  Measure,
  Source,
  Filters,
  WidgetSettings,
} from '../types';

// define a dataNode type
type dataNode = {
  id: string;
  type: string;
  config: WidgetConfig;
  node: InfiniNode<Transform>;
};

const _getDataType = (dataNodeType: string): QueryType => {
  switch (dataNodeType) {
    case 'PointMap':
      return QueryType.point;
    case 'GeoHeatMap':
      return QueryType.heat;
    case 'ChoroplethMap':
      return QueryType.choropleth;
    default:
      return QueryType.sql;
  }
};
const _getQueryParams = (dataNode: dataNode) => {
  const {type, config} = dataNode;
  const {width, height, pointSize, colorKey, bounds = {}, zoom = 5.36} = config;
  const {_sw = {}, _ne = {}} = bounds;
  const bounding_box = [_sw.lng, _sw.lat, _ne.lng, _ne.lat];
  let res = {
    width,
    height,
  };
  switch (_getDataType(type)) {
    case QueryType.point:
      return {
        ...res,
        point: {
          bounding_box,
          coordinate: 'EPSG:4326',
          stroke_width: pointSize,
          stroke: colorKey,
          opacity: 0.5,
        },
      };
    //TODO:
    case QueryType.heat:
      return {
        ...res,
        heat: {
          bounding_box,
          coordinate: 'EPSG:4326',
          map_scale: zoom,
        },
      };
    case QueryType.choropleth:
      return {
        ...res,
        choropleth: {
          bounding_box,
          coordinate: 'EPSG:4326',
          color_style: 'blue_to_red',
          rule: [2.5, 5],
          opacity: 1,
        },
      };
    default:
      return null;
  }
};

export const getDefaultConfig = (source: string, existLayouts: Layout[]): WidgetConfig => {
  let _id: string = id();
  return {
    id: _id,
    title: '',
    filter: {},
    selfFilter: {},
    type: DEFAULT_CHART,
    source,
    dimensions: [],
    measures: [],
    layout: {
      ...getInitLayout(existLayouts),
      i: _id,
    },
    colorKey: DEFAULT_COLOR,
    isServerRender: false,
  };
};

const _parseConfigToTransform = (config: WidgetConfig): Transform[] => {
  let transform: Transform[] = [];
  let aggTransform: Transform;
  let sortTransform: Transform;
  let limitTransform: Transform;

  config = cloneObj(config);
  // If no measure, put default measure for sql
  if (config.measures.length === 0) {
    config.measures.push({
      format: '',
      type: COLUMN_TYPE.NUMBER,
      label: '',
      expression: 'count',
      value: '*',
      as: 'countval',
    });
  }

  // agg
  let measures: any[] = config.measures.map((m: Measure) => ({
    type: m.expression,
    field: m.value,
    as: m.as,
  }));

  // non-bins groups
  const nonBinDimsExprs = config.dimensions
    .filter(d => !d.isBinned)
    .map(d => ({
      type: 'project',
      expr: d.expression ? {...d, type: d.expression} : d.value, //TODO:
      as: d.as,
    }));

  // bin groups
  let binDims = config.dimensions.filter(d => d.isBinned);
  // time bin groups
  let timeBinDims = binDims.filter(b => b.timeBin);
  // numeric bin groups
  const numericBinDimsExprs = binDims
    .filter(b => !b.timeBin)
    .map(b => {
      const {value, extent, maxbins = 0, as} = b;
      //alias, field, extent, maxbins
      return Helper.bin(as, value, extent as number[], maxbins);
    });

  const timeBinDimsExprs = timeBinDims.map(t => {
    if (t.extract) {
      return Helper.alias(t.as, t.timeBin==='isodow' ?`date_format(${t.value}, 'e')` :`${t.timeBin}(${t.value})`);
    }
    return Helper.alias(
      t.as,
      t.timeBin === 'day' ? `date(${t.value})` : `trunc(${t.value}, '${t.timeBin}')`
    );
  });

  const hasAggregate =
    nonBinDimsExprs.length + timeBinDimsExprs.length + numericBinDimsExprs.length > 0;

  if (hasAggregate) {
    // transform agg
    aggTransform = Helper.aggregate([...nonBinDimsExprs, ...timeBinDimsExprs], measures);
    // push num bin groups
    Array.isArray(aggTransform.groupby) && aggTransform.groupby.push(...numericBinDimsExprs);
    // add to transform
    transform.push(aggTransform);
  } else {
    measures.forEach((m: Measure) => {
      transform.push({
        type: 'project',
        expr: m,
        as: m.as,
      });
    });
  }

  // if we have sort, add sort transform
  if (config.sort) {
    sortTransform = Helper.sort(config.sort.name, config.sort.order);
    transform.push(sortTransform);
  }

  // for limit and offset
  if (typeof config.limit === 'number') {
    limitTransform = Helper.limit(config.limit!, config.offset || 0);
    transform.push(limitTransform);
  }

  // just resolve crossfilter , it won't affact others
  let xFilterExpr: ResolveFilter = {
    type: 'resolvefilter',
    filter: {signal: 'crossfilter'},
  };
  if (config.ignore) {
    xFilterExpr.ignore = config.ignore.map((v: string) =>
      prefixFilter(config.ignoreId || config.id, v)
    );
  }
  transform.push(xFilterExpr);
  // console.info(transform);
  return transform;
};

export const prefixFilter = (prefix: string, name: string): string => {
  return `${prefix}_${name}`;
};

// create cross filter and generate sql for each widget
export const getWidgetSql = (
  configs: WidgetConfig[],
  sources: Source[] = [],
  widgetSettings: WidgetSettings
) => {
  // create a config map
  const configMap: Map<string, WidgetConfig> = new Map();
  // add count table
  const countTable = sources.map((s: Source) => {
    return {id: s, type: 'count', source: s, dimensions: [], measures: []};
  });
  // get all configs required for generating sql
  const copiedConfigs = [...countTable, ...cloneObj(configs)];

  // fill the config map
  copiedConfigs.forEach((config: WidgetConfig) => {
    // get widget setting
    const widgetSetting = widgetSettings[config.type];
    // pre process config if needed
    const processedConfig = widgetSetting ? widgetSetting.configHandler(config) : config;
    // store the processed config in the map
    configMap.set(config.id, processedConfig);
    // special for range chart
    if (config.isShowRange) {
      // get range chart config
      let rangeConfig = rangeConfigGetter(config);
      // store in the map
      configMap.set(rangeConfig.id, widgetSetting.configHandler(rangeConfig));
    }
  });
  // create a Query array
  let widgetQuerys: Query[] = [];
  // create a root data graph
  const collection = new InfiniCollection<Transform>({
    reducer: SqlParser.reducer,
    stringify: SqlParser.toSQL,
  });
  // create a map for save all cross filtering nodes
  // widget have the same source will have the same cross filtering mother
  let xfilterNodes = new Map<string, InfiniNode<Transform>>();
  // create a empty data nodes array
  let dataNodes: dataNode[] = [];

  // loop our widget's configs
  configMap.forEach((config: WidgetConfig) => {
    // get transform for the config
    let transform = _parseConfigToTransform(config);
    // get current cross filtering node by source
    let xfilterNode: InfiniNode<Transform> = xfilterNodes.get(config.source)!;
    // if we have the cross filtering node set, get it
    // otherwise, create a new one
    if (!xfilterNodes.has(config.source)) {
      xfilterNode = collection.create(config.source, [
        {
          type: 'crossfilter',
          signal: 'crossfilter',
          filter: {},
        },
      ]);
      xfilterNodes.set(config.source, xfilterNode);
    }
    // create data node, every widget is a data node and it is a child node of the cross filtering node;
    dataNodes.push({
      id: config.id,
      type: config.type,
      config: config,
      node: xfilterNode.add({
        transform: transform,
      }),
    });

    // get all filters
    const configFilters: Filters = config.filter || config.selfFilter;
    // if we got filter, set it in the cross filtering node as global filter first
    if (configFilters) {
      xfilterNode.setTransform((transforms: Transform[]) => {
        let crossfilters: Crossfilter = transforms[0] as Crossfilter;
        let filters: Filters = {};
        Object.keys(config.filter || {}).forEach((f: string) => {
          // make sure we can have same filter name in any chart
          // so we add a random prefix
          filters[prefixFilter(config.id, f)] = configFilters[f];
        });
        // set the cross filtering node's filter
        crossfilters.filter = {...crossfilters.filter, ...filters};
        return transforms;
      });
    }
  });

  // return sql
  dataNodes.forEach((dataNode: dataNode) => {
    dataNode.node.setTransform((transforms: Transform[]) => {
      // add selfFilter to every dataNode itself
      if (dataNode.config.selfFilter) {
        Object.keys(dataNode.config.selfFilter || {}).forEach((key: string) => {
          transforms.push(dataNode.config.selfFilter[key]);
        });
      }
      return transforms;
    });
    // preparation before removing current filter
    const config = dataNode.config;
    const filter = config.filter;
    let xfilterNode: InfiniNode<Transform> = xfilterNodes.get(config.source)!;
    let tempCrossfilters: Filters;
    const configFilterNames = Object.keys(filter || {}).map((v: string) =>
      prefixFilter(config.id, v)
    );
    const keepFilter = (config.keep || []).map((v: string) => prefixFilter(config.id, v));
    const hasFilter = Object.keys(config.filter || {}).length > 0;

    // remove current filter from global filter when gen sql for each dataNode
    if (!config.isServerRender && hasFilter) {
      xfilterNode.setTransform((transforms: Transform[]) => {
        let crossfilters: Crossfilter = transforms[0] as Crossfilter;
        tempCrossfilters = cloneObj(crossfilters.filter);
        let xfilters = Object.keys(tempCrossfilters);
        xfilters.forEach((x: string) => {
          let shouldNotRemove = keepFilter.indexOf(x) !== -1;
          let hasXfilter = configFilterNames.indexOf(x) !== -1;
          if (!shouldNotRemove && hasXfilter) {
            delete crossfilters.filter[x];
          }
        });
        return transforms;
      });
    }

    const widgetSetting = widgetSettings[config.type];

    // generate sql
    let sql = dataNode.node.reduceToString();
    if (widgetSetting && widgetSetting.onAfterSqlCreate) {
      sql = widgetSetting.onAfterSqlCreate(sql, config);
    }
    const type = _getDataType(dataNode.type);
    const params = _getQueryParams(dataNode);
    const query: any = {sql, type};
    if (type !== QueryType.sql) {
      query.params = params;
    }
    widgetQuerys.push({
      id: dataNode.id,
      params: query,
    });
    if (!config.isServerRender && hasFilter) {
      // recover current fitler
      xfilterNode.setTransform((transforms: Transform[]) => {
        let crossfilters: Crossfilter = transforms[0] as Crossfilter;
        crossfilters.filter = tempCrossfilters;
        return transforms;
      });
    }
  });
  return widgetQuerys;
};
