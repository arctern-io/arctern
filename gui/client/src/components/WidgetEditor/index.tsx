import React, {FC, useContext, useState, useEffect, useRef, Suspense} from 'react';
import * as customSetting from '../settingComponents';
import {useTheme, makeStyles} from '@material-ui/core/styles';
import {I18nContext} from '../../contexts/I18nContext';
import {WidgetEditorProps} from '../../types/Editor';
import Source from './Source';
import Dimensions from './Dimensions';
import Measures from './Measures';
import WidgetEditorContent from './WidgetEditorContent';
import Spinner from '../common/Spinner';
import {cloneObj} from '../../utils/Helpers';
import {isReadyToRender, genEffectClickOutside} from '../../utils/EditorHelper';
import {genWidgetEditorStyle, genCustomSettingStyle} from './index.style';
import './index.scss';

// dynamic components cache;
const cache = new Map();

const useStyles = makeStyles(theme => genWidgetEditorStyle(theme) as any) as Function;
const useCustomSettingStyles = makeStyles(theme => genCustomSettingStyle(theme) as any) as Function;

const WidgetEditor: FC<WidgetEditorProps> = props => {
  const {nls} = useContext(I18nContext);
  const theme = useTheme();
  const classes = useStyles(theme);
  const customSettingClasses = useCustomSettingStyles(theme);
  const {config, setConfig, dashboard, dataMeta, data, setting} = props;
  const {sources, sourceOptions} = dashboard;
  const cloneConfig = cloneObj(config);
  const {source = '', type = ''} = cloneConfig;
  const opts = sourceOptions[source];
  const [status, setStatus] = useState('showChart'); // "showChart" | [sourceVals]
  const [[width, height], setChartSize] = useState([-1, -1]);
  const chartContainer = useRef<HTMLDivElement>(null);
  const baseInfoNode = useRef<HTMLElement>(null);
  // Dynamic load chart settings and view components
  // let CustomEditComponent;
  let chartKey = `${type}/view`;
  let chartEditorKey = `${type}/CustomEditor`;
  let Widget = cache.get(chartKey);
  let CustomEditComponent = cache.get(chartEditorKey);
  // load chart component if not existing
  Widget = Widget || React.lazy(() => import(`../../widgets/${chartKey}`));
  // load chart edit component if not existing
  CustomEditComponent =
    CustomEditComponent || React.lazy(() => import(`../../widgets/${chartEditorKey}`));
  // update the cache
  cache.set(chartKey, Widget);
  cache.set(chartEditorKey, CustomEditComponent);

  // caculate if we can render the chart
  const {sourceReady, dimensionsReady, measuresReady} = isReadyToRender(config, setting);
  const isConfigReady = sourceReady.isReady && dimensionsReady.isReady && measuresReady.isReady;
  const isReady = isConfigReady && width > 0 && height > 0;
  const effectFactors = JSON.stringify([
    chartContainer.current && chartContainer.current.getBoundingClientRect(),
    isConfigReady,
  ]);
  // set Chart's height and width
  useEffect(() => {
    if (chartContainer.current) {
      const {width, height} = chartContainer.current.getBoundingClientRect();
      const sideWidth = 600;
      const paddingBottom = 40;
      const titleHeight = 30;

      if (width > 0 && status === 'showChart') {
        setChartSize([width - sideWidth, height - titleHeight - paddingBottom]);
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [effectFactors]);

  // change to chart mode when clickOutside
  useEffect(() => {
    const changtoShowChart = genEffectClickOutside(baseInfoNode.current, setStatus, 'showChart');
    document.body.addEventListener('click', changtoShowChart);
    return () => {
      document.body.removeEventListener('click', changtoShowChart);
    };
  }, [status]);

  const onMouseEnterSourceOption = (val: string) => {
    setStatus(val);
  };

  return (
    <Suspense fallback={<Spinner />}>
      <div className={classes.root}>
        <div className={classes.sidebar}>
          <div className={classes.sources}>
            <div className={classes.title}>{nls.label_sources}</div>
            <Source
              config={config}
              setConfig={setConfig}
              options={sources}
              onMouseOver={onMouseEnterSourceOption}
            />
          </div>
          <div className={`${classes.sources} ${classes.sidebarLeft}`}>
            <div className={classes.title}>{nls.label_dimensions}</div>
            <Dimensions
              config={config}
              setConfig={setConfig}
              dimensionsSetting={setting && setting.dimensions}
              options={opts}
            />
          </div>
          <div className={`${classes.sources}`}>
            <div className={classes.title}>{nls.label_measures}</div>
            <Measures
              {...props}
              config={config}
              setConfig={setConfig}
              measuresSetting={setting && setting.measures}
              options={opts}
            />
          </div>
        </div>
        <div className={classes.chart} ref={chartContainer}>
          <WidgetEditorContent
            {...props}
            status={status}
            isReady={isReady}
            dimensionsReady={dimensionsReady}
            measuresReady={measuresReady}
            Widget={Widget}
            width={width}
            height={height}
          />
        </div>

        <div className={`${classes.sidebar} ${classes.sidebarRight}`}>
          <CustomEditComponent
            {...customSetting}
            classes={customSettingClasses}
            config={config}
            setConfig={setConfig}
            nls={nls}
            options={opts}
            dataMeta={dataMeta}
            data={data}
          />
        </div>
      </div>
    </Suspense>
  );
};

export default WidgetEditor;
