import React, {FC, Suspense, useState, useRef, useEffect, useContext, useReducer} from 'react';
import clsx from 'clsx';
import {makeStyles, useTheme} from '@material-ui/core/styles';
import {
  Settings as SettingsIcon,
  Clear as ClearIcon,
  SaveAlt as SaveAltIcon,
} from '@material-ui/icons';
import WidgetEditor from '../../components/WidgetEditor';
import {DataQuery, getLinkData} from '../../utils/Query';
import {
  DefaultWidgetProps,
  WidgetConfig,
  Query,
  Data,
  Meta,
  Dimension,
  Measure,
  DataCache,
} from '../../types';
import {MODE, CONFIG} from '../../utils/Consts';
import Header from './Header';
import Spinner from '../../components/common/Spinner';
import {genWidgetWrapperStyle} from './WidgetWrapper.style';
import {cloneObj} from '../../utils/Helpers';
import {getWidgetSql} from '../../utils/Configs';
import {exportCsv} from '../../utils/Export';
import {isReadyToRender} from '../../utils/EditorHelper';
import {queryContext} from '../../contexts/QueryContext';
import {I18nContext} from '../../contexts/I18nContext';
import {rootContext} from '../../contexts/RootContext';
import localConfigReducer from '../../utils/reducers/localConfigReducer';
import {getWidgetTitle} from '../../utils/WidgetHelpers';

// component cache
const widgetsMap = new Map();
const useStyles = makeStyles(theme => genWidgetWrapperStyle(theme) as any) as any;
const _deleteFormat = (items: Array<Dimension | Measure>) => {
  if (items.length === 0) {
    return items;
  }
  return items.map((item: Dimension | Measure) => {
    delete item.format;
    return item;
  });
};
const _deleteUnRelatedAttr = (config: WidgetConfig) => {
  if (!config) {
    return config;
  }
  const cloneConfig = cloneObj(config);
  const {dimensions = [], measures = []} = cloneConfig;
  cloneConfig.dimensions = _deleteFormat(dimensions);
  cloneConfig.measures = _deleteFormat(measures);
  if (!cloneConfig.isServerRender) {
    delete cloneConfig.ruler;
  }
  delete cloneConfig.rulerBase;
  return cloneConfig;
};
const WidgetWrapper: FC<DefaultWidgetProps> = props => {
  const {getData} = useContext(queryContext);
  const {nls} = useContext(I18nContext);
  const theme = useTheme();
  const {widgetSettings} = useContext(rootContext);
  const classes = useStyles();
  const container = useRef<HTMLDivElement>(null);
  const {
    dashboard,
    config,
    data,
    linkData,
    mode,
    dataMeta,
    linkMeta,
    setMode,
    setConfig,
    configs,
  } = props;
  const dataCache = useRef<DataCache>({});
  const dataQueryCache = useRef<DataQuery>(
    new DataQuery({
      requester: getData,
      onRequest: (query: Query) => {
        setLocalMeta((meta: Meta) => {
          const copiedMeta = cloneObj(meta);
          const {params, id, timestamp} = query;
          copiedMeta[query.id] = {params, id, timestamp, loading: true};
          return copiedMeta;
        });
      },
      onResponse: (query: Query, data: Data) => {
        dataCache.current[query.id] = data;
        setLocalMeta((meta: Meta) => {
          const {params, id, timestamp} = query;
          const copiedMeta = cloneObj(meta);
          copiedMeta[id] = {params, id, timestamp, loading: false};
          return copiedMeta;
        });
      },
    })
  );
  const [localMeta, setLocalMeta] = useState<Meta>({});
  const [isHover, setHover] = useState<boolean>(false);
  const [localConfig, _setLocalConfig] = useReducer(localConfigReducer, cloneObj(config));

  const calSize = () => {
    const wrapper = container.current || document.createElement('div');
    const {width, height} =
      {width: wrapper.offsetWidth, height: wrapper.offsetHeight} || wrapper.getBoundingClientRect();
    const headerHeight = 40;
    return [width, height - headerHeight];
  };

  const [[clientWidth, clientHeight], setClientSize]: any = useState([0, 0]);

  useEffect(() => {
    const resizeClientSize = () => {
      setClientSize(calSize());
    };
    window && window.addEventListener('resize', resizeClientSize);
    return () => window && window.removeEventListener('resize', resizeClientSize);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    const [width, height] = calSize();
    if (width <= 0 || height <= 0) {
      return;
    }
    setClientSize([width, height]);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [JSON.stringify(config.layout)]);

  // widget mode
  let widgetMode: MODE = mode.mode === MODE.EDIT && config.id === mode.id ? MODE.EDIT : MODE.NORMAL;
  // flags
  const isEdit: Boolean = widgetMode === MODE.EDIT;
  const isAdd: Boolean = mode.mode === MODE.ADD;
  const useLocalState: Boolean = isEdit || isAdd;

  // new configs with local config merged
  const localConfigs: WidgetConfig[] = cloneObj(configs);
  const currentIndex = localConfigs.findIndex((c: WidgetConfig) => c.id === config.id);
  localConfigs.splice(currentIndex, 1, localConfig);

  // add and edit are all belong to edit mode
  widgetMode = isAdd ? MODE.EDIT : widgetMode;
  // dynamic load widget component, so that we don't have to import all kinds of widget
  let Widget;
  let widgetKey = `${config.type}/widget`;
  let widgetType = useLocalState ? localConfig.type : config.type;
  if (widgetsMap.has(widgetKey)) {
    Widget = widgetsMap.get(widgetKey);
  } else {
    Widget = React.lazy(() => import(`../../widgets/${widgetType}/view`));
    widgetsMap.set(widgetKey, Widget);
  }

  // clear cache
  useEffect(() => {
    dataCache.current = {};
    setLocalMeta({});
  }, [config.type, localConfig.type]);

  // avoid multiple requests
  const queryTimeout = useRef<any>(null);
  const isFilterExist = !!Object.keys(config.filter).length;
  // only fire on useLocalState
  useEffect(() => {
    if (!useLocalState) {
      return;
    }
    // check if we can make the rquest
    const {sourceReady, dimensionsReady, measuresReady} = isReadyToRender(
      localConfig,
      widgetSettings[localConfig.type]
    );
    if (sourceReady.isReady && dimensionsReady.isReady && measuresReady.isReady) {
      // generate sql
      const querys = getWidgetSql(localConfigs, [], widgetSettings).filter(
        (c: any) => c.id === localConfig.id || c.id === localConfig.linkId
      );

      if (queryTimeout.current) {
        clearTimeout(queryTimeout.current);
      }
      // throttle requests
      queryTimeout.current = setTimeout(() => {
        dataQueryCache.current.q(querys);
      }, 16);
    }
    // get new data using updated nodes
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [JSON.stringify(_deleteUnRelatedAttr(localConfig))]);

  // compute data
  const localData = dataCache.current[localConfig.id] || [];
  const localLinkData = getLinkData(dataCache.current, localConfig);

  const onMouseEnter = () => {
    setHover(true);
  };

  const onMouseLeave = () => {
    setHover(false);
  };

  // clear filter
  const onChartClearFilter = () => {
    setConfig({
      type: CONFIG.CLEAR_FILTER,
      payload: config,
    });
  };

  // go to widget setting
  const handleSettingCLick = () => {
    setMode({mode: MODE.EDIT, id: config.id});
  };

  // delete widget
  const delWidget = () => {
    setConfig({type: CONFIG.DEL, payload: config});
  };

  // export widget data to csv
  const handleCSVExport = () => {
    exportCsv(config, data);
  };

  // uniform config setter interface
  const setWidgetConfig = (props: any, useNormal: boolean) => {
    if (useLocalState && !useNormal) {
      _setLocalConfig({type: props.type || CONFIG.UPDATE, payload: props.payload});
    } else {
      setConfig(props);
    }
  };

  return (
    <>
      <Suspense fallback={<Spinner />}>
        {widgetMode === MODE.NORMAL && (
          <div
            ref={container}
            className={classes.container}
            onMouseEnter={onMouseEnter}
            onMouseLeave={onMouseLeave}
          >
            <div className={classes.header}>
              <h3>{getWidgetTitle(config, nls)}</h3>
              <div className={clsx(classes.actions, isHover ? '' : classes.hidden)}>
                <span className={`${classes.link} ${isFilterExist ? '' : classes.hidden}`}>
                  <div
                    className={classes.icon}
                    onClick={onChartClearFilter}
                    dangerouslySetInnerHTML={{
                      __html: `<svg class="icon" viewBox="0 0 48 48"><polygon style="fill: ${theme.palette.primary.main}" points="46,29.9 44.1,28 40,32.2 35.9,28 34,29.9 38.2,34 34,38.1 35.9,40 40,35.8 44.1,40 46,38.1 41.8,34  "></polygon><g id="icon-filter"><path fill=${theme.palette.primary.main} d="M40,6.5H8L6.8,8.9l11.7,15.6V44v2.4l2.2-1.1l0,0l8-4l0,0l0.8-0.4V40V24.5L41.2,8.9L40,6.5z M25,23v16.8l-3.5,1.8V24v-0.5 l-0.3-0.4l0,0l-5.4-7.3L11,9.5h24.1L25,23z"></path></g></svg>`,
                    }}
                  />
                </span>
                <span className={classes.link}>
                  <SettingsIcon classes={{root: classes.hover}} onClick={handleSettingCLick} />
                </span>
                {!config.isServerRender && (
                  <span className={classes.link}>
                    <SaveAltIcon onClick={handleCSVExport} />
                  </span>
                )}
                <span className={classes.link}>
                  <ClearIcon onClick={delWidget} />
                </span>
              </div>
            </div>
            {dataMeta && dataMeta.loading && (
              <div className="loading-container">
                <Spinner delay={1000} />
              </div>
            )}
            {clientWidth > 0 && clientHeight > 0 && (
              <Widget
                setConfig={setConfig}
                config={config}
                data={data}
                linkData={linkData}
                dataMeta={dataMeta}
                linkMeta={linkMeta}
                wrapperWidth={clientWidth}
                wrapperHeight={clientHeight}
              />
            )}
          </div>
        )}

        {widgetMode === MODE.EDIT && (
          <div className={classes.container}>
            {useLocalState && (
              <Header
                mode={mode}
                setMode={setMode}
                config={config}
                dashboard={dashboard}
                localConfig={localConfig}
                setConfig={setWidgetConfig}
                widgetSettings={widgetSettings}
              />
            )}
            <WidgetEditor
              setting={widgetSettings[localConfig.type]}
              dashboard={dashboard}
              config={localConfig}
              setConfig={setWidgetConfig}
              data={localData}
              linkData={localLinkData}
              dataMeta={localMeta[localConfig.id]}
              linkMeta={localMeta[localConfig.linkId!]}
              wrapperWidth={clientWidth}
              wrapperHeight={clientHeight}
            />
          </div>
        )}
      </Suspense>
    </>
  );
};

export default WidgetWrapper;
