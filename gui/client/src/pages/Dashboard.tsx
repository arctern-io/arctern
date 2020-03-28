import React, {FC, useState, useContext, useReducer, useEffect, useRef} from 'react';
import RGL, {WidthProvider} from 'react-grid-layout';
import {makeStyles, useTheme} from '@material-ui/core/styles';
import Header from './containers/Header';
import WidgetWrapper from './containers/WidgetWrapper';
import {rootContext} from '../contexts/RootContext';
import {queryContext} from '../contexts/QueryContext';
import EmptyChart from '../components/common/EmptyWidget';
import {getDefaultConfig, getWidgetSql} from '../utils/Configs';
import {DataQuery, getLinkData} from '../utils/Query';
import {cloneObj} from '../utils/Helpers';
import {MODE, DASH_ACTIONS} from '../utils/Consts';
import {fullLayoutWidth, fullLayoutHeight} from '../utils/Layout';
import configsReducer from '../utils/reducers/configsReducer';
import {WidgetConfig, DashboardProps, Mode, Layout, Query, Data, DataCache, Meta} from '../types';
import 'react-grid-layout/css/styles.css';
import './Dashboard.scss';
const useStyles = makeStyles(theme => ({
  root: {
    flexGrow: 1,
  },
  wrapper: {
    display: 'flex',
    backgroundColor: theme.palette.background.default,
  },
}));
// const ResponsiveGridLayout = WidthProvider(Responsive);
const ReactGridLayout = WidthProvider(RGL);
const _getLayouts = (configs: WidgetConfig[]) =>
  configs.map((config: WidgetConfig) => config.layout);
// core component
const Dashboard: FC<DashboardProps> = ({dashboard, setDashboard}) => {
  const {configs, id, demo, sources} = dashboard;
  const {getData, isFirefox} = useContext(queryContext);
  const {widgetSettings} = useContext(rootContext);
  const theme = useTheme();
  // get sourceOptions like dimensions options, measures options
  const classes = useStyles(theme);
  const [meta, setMeta] = useState<Meta>({});
  const [sourceData, setSourceData] = useState<DataCache>({});
  const dataCache = useRef<DataCache>({});
  const dataQueryCache = useRef<DataQuery>(
    new DataQuery({
      requester: getData,
      onRequest: (query: Query) => {
        setMeta((meta: Meta) => {
          const copiedMeta = cloneObj(meta);
          const {params, id} = query;
          if (sources.includes(id)) {
            return meta;
          }
          copiedMeta[query.id] = {params, id, loading: true};
          return copiedMeta;
        });
      },
      onResponse: (query: Query, data: Data) => {
        dataCache.current[query.id] = data;
        setMeta((meta: Meta) => {
          const copiedMeta = cloneObj(meta);
          const {params, id} = query;
          if (sources.includes(id)) {
            setSourceData((prev: any) => ({...prev, [id]: data}));
            return meta;
          }
          copiedMeta[query.id] = {params, id, loading: false};
          return copiedMeta;
        });
      },
    })
  );

  // Edit mode or normal mode
  const [mode, setMode] = useState<Mode>({mode: MODE.NORMAL, id: ''});
  const isEdit = mode.mode === MODE.ADD || mode.mode === MODE.EDIT;
  const isAdd = mode.mode === MODE.ADD;
  // const [rowHeight, setRowHeight] = useState(36);

  // configs reducer
  const [widgetConfigs, setConfig] = useReducer(configsReducer, [...configs]);

  // avoid dobule requests on useEffect at first run
  const isFirstRun = useRef(true);
  // widget dispatch filter or other state change
  // => configsReducer generate params with new sql
  // => send request to backend
  useEffect(() => {
    if (isFirstRun.current) {
      isFirstRun.current = false;
    }
    // parse configs to querys, create cross filter nodes and sqls
    let querys = getWidgetSql(widgetConfigs, sources, widgetSettings);
    // send to the backend
    dataQueryCache.current.q(querys);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [JSON.stringify(widgetConfigs), id]);

  // dispatch configs changes to parent
  useEffect(() => {
    if (isFirstRun.current) {
      return;
    }
    setDashboard({type: DASH_ACTIONS.UPDATE_CONFIGS, payload: widgetConfigs});
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [JSON.stringify(widgetConfigs)]);

  const onLayoutChange = (layouts: Layout[]) => {
    const isLayoutChange = JSON.stringify(layouts) !== JSON.stringify(_getLayouts(widgetConfigs));
    if (isLayoutChange) {
      setConfig({
        type: DASH_ACTIONS.LAYOUT_CHANGE,
        payload: layouts,
      });
    }
  };

  const onDragStart = function(...args: any) {
    let evt = args[4];
    let target = evt.target;
    let isCanvas = target && target.tagName === 'CANVAS';
    let isLoading = target && target.classList.contains('loading-container');
    return !isCanvas && !isLoading;
  };
  if (isEdit) {
    const config = widgetConfigs.filter((config: WidgetConfig) => config.id === mode.id)[0];
    const dataMeta = config && meta[config.id];
    const data = config && dataCache.current[config.id];
    return (
      <WidgetWrapper
        mode={mode}
        setMode={setMode}
        config={isAdd ? getDefaultConfig(sources[0], _getLayouts(widgetConfigs)) : config}
        data={data || []}
        dataMeta={dataMeta || {}}
        configs={widgetConfigs}
        setConfig={setConfig}
        dashboard={dashboard}
      />
    );
  }
  return (
    <div className={classes.root}>
      <Header
        showRestoreBtn={!demo}
        mode={mode}
        setMode={setMode}
        dashboard={dashboard}
        setDashboard={setDashboard}
        setConfig={setConfig}
        configs={widgetConfigs}
        data={sourceData}
      />
      {widgetConfigs.length > 0 && (
        <div style={{padding: '0 0 0 24px'}}>
          <ReactGridLayout
            cols={fullLayoutWidth}
            rowHeight={fullLayoutHeight}
            isDraggable={true}
            onDragStart={onDragStart}
            layout={_getLayouts(widgetConfigs)}
            onLayoutChange={onLayoutChange}
            verticalCompact={true}
            useCSSTransforms={!isFirefox}
            margin={[theme.spacing(1), theme.spacing(1)]}
            // containerPadding={[0, 0]}
          >
            {widgetConfigs.map((config: WidgetConfig) => {
              return (
                <div key={config.id} className={classes.wrapper}>
                  <WidgetWrapper
                    isLoading={false}
                    mode={mode}
                    config={config}
                    configs={widgetConfigs}
                    setConfig={setConfig}
                    setMode={setMode}
                    dashboard={dashboard}
                    data={dataCache.current[config.id] || []}
                    dataMeta={meta[config.id]}
                    linkMeta={meta[config.linkId!]}
                    linkData={getLinkData(dataCache.current, config) || []}
                  />
                </div>
              );
            })}
          </ReactGridLayout>
        </div>
      )}

      {widgetConfigs.length === 0 && <EmptyChart setMode={setMode} />}
    </div>
  );
};

export default Dashboard;
