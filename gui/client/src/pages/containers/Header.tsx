import React, {useState, useContext, useEffect, useRef, FC} from 'react';
import clsx from 'clsx';
import {makeStyles, useTheme} from '@material-ui/core/styles';
import {
  GridOn as GridOnIcon,
  ViewModule as ViewModuleIcon,
  ViewCompact as ViewCompactIcon,
  RestorePage as RestorePageIcon,
  ViewArray as ViewArrayIcon,
  ViewQuilt as ViewQuiltIcon,
} from '@material-ui/icons';
import {Button, Card} from '@material-ui/core';
import WidgetSelector from '../../components/common/WidgetSelector';
import SourceStats from '../../components/common/SourceStats';
import EditableLabel from '../../components/common/EditableLabel';
import {I18nContext} from '../../contexts/I18nContext';
import {rootContext} from '../../contexts/RootContext';
import {HeaderProps, WidgetConfig} from '../../types';
import {isReadyToRender, convertConfig, genEffectClickOutside} from '../../utils/EditorHelper';
import {applyUsedLayout} from '../../utils/Layout';
import {MODE, CONFIG, DASH_ACTIONS} from '../../utils/Consts';
import {namespace} from '../../utils/Helpers';
import {getFilterLength} from '../../utils/Filters';
import {genHeaderStyle} from './Header.style';
const useStyles = makeStyles(theme => genHeaderStyle(theme) as any) as Function;
const Header: FC<HeaderProps> = props => {
  const theme = useTheme();
  const classes = useStyles(theme);
  const {nls} = useContext(I18nContext);
  const {widgetSettings} = useContext(rootContext);
  const {
    data,
    mode,
    config,
    localConfig = {
      id: '',
      source: '',
      dimensions: [],
      measures: [],
      type: '',
      filter: {},
      selfFilter: {},
      layout: {i: 'string', x: 1, y: 1, w: 10, h: 10, static: false},
    },
    setMode,
    setConfig,
    dashboard,
    setDashboard,
    configs = [],
    showRestoreBtn = true,
  } = props;

  const {id, sources, totals, title} = dashboard;

  let widgetMode: MODE = mode.mode === MODE.EDIT ? MODE.EDIT : MODE.NORMAL;
  const isEdit: Boolean = widgetMode === MODE.EDIT;
  const isAdd: Boolean = mode.mode === MODE.ADD;
  // add and edit are all belong to edit mode
  widgetMode = isAdd ? MODE.EDIT : widgetMode;
  const isNormal = widgetMode === MODE.NORMAL;
  const [isReady, setIsReady] = useState(false);
  const [showLayout, setShowLayout] = useState(false);
  const filtersLength = getFilterLength(configs.map((c: WidgetConfig) => c.filter));
  const cardRef = useRef(null);

  useEffect(() => {
    if (!localConfig.type) {
      setIsReady(true);
      return;
    }
    const {sourceReady, dimensionsReady, measuresReady} = isReadyToRender(
      localConfig,
      widgetSettings[localConfig.type]
    );
    const _isReady = sourceReady.isReady && dimensionsReady.isReady && measuresReady.isReady;
    setIsReady(_isReady);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [localConfig]);

  useEffect(() => {
    const card = cardRef.current || document.createElement('div');
    const _hideLayoutSetting = genEffectClickOutside(card, setShowLayout, false);
    document.body.addEventListener('click', _hideLayoutSetting);
    return () => document.body.removeEventListener('click', _hideLayoutSetting);
  }, [classes.hidden]);

  const showLayoutSettings = () => {
    setShowLayout(true);
  };

  const adaptLayout = (layoutType: string) => {
    const newConfigs = applyUsedLayout(configs, layoutType);
    setConfig({
      type: DASH_ACTIONS.CURR_USED_LAYOUT_CHANGE,
      payload: newConfigs,
    });
    setShowLayout(false);
  };

  const onCancel = () => {
    setConfig && setConfig({payload: {...config}});
    setMode({mode: MODE.NORMAL});
  };

  const onSave = () => {
    // dispatch config UPDATE
    setConfig({type: CONFIG.UPDATE, payload: localConfig}, true);
    setMode({mode: MODE.NORMAL});
  };

  const onAdd = () => {
    // init node
    // dispatch config ADD
    setConfig({type: CONFIG.UPDATE, payload: localConfig}, true);
    setMode({mode: MODE.NORMAL});
  };

  const onClearFilter = () => {
    setConfig({type: CONFIG.CLEARALL_FILTER, payload: configs});
  };

  const onWidgetTypeChange = (widgetType: string, _localConfig: any = localConfig) => {
    const _config = convertConfig(_localConfig, widgetType);
    setConfig && setConfig({payload: _config, type: CONFIG.REPLACE_ALL});
  };

  const onRestoreLayout = () => {
    window.localStorage.removeItem(namespace(['dashboard'], String(id)));
    window.location.reload();
  };

  const onTitleChange = ({title}: any) => {
    if (setDashboard) {
      setDashboard({type: DASH_ACTIONS.UPDATE_TITLE, payload: title});
    }
  };

  // should we disable the apply button
  const shouldBtnEnabled = isReady && JSON.stringify(config) !== JSON.stringify(localConfig);
  return (
    <>
      {isNormal && (
        <div className={classes.root}>
          <div className={classes.title}>
            <EditableLabel onChange={onTitleChange} label={title} />
            <SourceStats data={data!} sources={sources} totals={totals} />
          </div>
          <div className={classes.editor}>
            <Button
              className={clsx(classes.hover, classes.marginRight)}
              variant="contained"
              size="small"
              color="primary"
              onClick={() => {
                setMode({mode: MODE.ADD});
              }}
            >
              {`+ ${nls.label_add_new_widget}`}
            </Button>
            <div className={classes.tools}>
              <div className={classes.tool}>
                <Button
                  variant="contained"
                  className={classes.hover}
                  classes={{root: classes.layout}}
                  onClick={showLayoutSettings}
                >
                  <GridOnIcon />
                </Button>
                <Card
                  ref={cardRef}
                  classes={{
                    root: clsx(showLayout ? '' : classes.hidden, classes.layoutTypes),
                  }}
                >
                  <ViewQuiltIcon
                    classes={{root: classes.hover}}
                    onClick={() => {
                      adaptLayout('_4211');
                    }}
                  />
                  <ViewModuleIcon
                    classes={{root: classes.hover}}
                    onClick={() => {
                      adaptLayout('_9avg');
                    }}
                  />
                  <ViewQuiltIcon
                    classes={{root: clsx(classes.hover, classes.transform)}}
                    onClick={() => {
                      adaptLayout('_1124');
                    }}
                  />
                  <ViewCompactIcon
                    classes={{root: classes.hover}}
                    onClick={() => {
                      adaptLayout('_timelineTop');
                    }}
                  />
                  <ViewArrayIcon
                    classes={{root: classes.hover}}
                    onClick={() => {
                      adaptLayout('_mapdDashboard');
                    }}
                  />
                </Card>
              </div>
              <div
                className={clsx(classes.fiterClear, classes.tool, classes.hover)}
                onClick={onClearFilter}
              >
                <Button
                  variant="contained"
                  className={classes.hover}
                  classes={{root: classes.layout}}
                >
                  <div
                    className={classes.icon}
                    dangerouslySetInnerHTML={{
                      __html: `<svg class="icon" viewBox="0 0 48 48"><polygon style="fill: currentColor" points="46,29.9 44.1,28 40,32.2 35.9,28 34,29.9 38.2,34 34,38.1 35.9,40 40,35.8 44.1,40 46,38.1 41.8,34  "></polygon><g id="icon-filter"><path fill=currentColor d="M40,6.5H8L6.8,8.9l11.7,15.6V44v2.4l2.2-1.1l0,0l8-4l0,0l0.8-0.4V40V24.5L41.2,8.9L40,6.5z M25,23v16.8l-3.5,1.8V24v-0.5 l-0.3-0.4l0,0l-5.4-7.3L11,9.5h24.1L25,23z"></path></g></svg>`,
                    }}
                  />
                </Button>
                <span className={`${classes.filterNum} ${filtersLength > 0 ? '' : classes.hidden}`}>
                  {filtersLength}
                </span>
              </div>
              {showRestoreBtn && (
                <div className={classes.tool}>
                  <Button
                    variant="contained"
                    className={classes.hover}
                    classes={{root: classes.layout}}
                    onClick={onRestoreLayout}
                  >
                    <RestorePageIcon />
                  </Button>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {(isEdit || isAdd) && (
        <div className={classes.editHeader}>
          <Button variant="outlined" color="default" onClick={onCancel}>
            {nls.label_cancel}
          </Button>
          {Object.keys(widgetSettings).map((widgetType: any) => {
            const settings = widgetSettings[widgetType];
            return (
              <WidgetSelector
                key={widgetType}
                icon={settings.icon}
                widgetType={widgetType}
                selected={widgetType === (localConfig && localConfig.type)}
                onClick={onWidgetTypeChange}
              />
            );
          })}
          <Button
            variant="contained"
            disabled={!shouldBtnEnabled}
            color="primary"
            onClick={isReady ? (isAdd ? onAdd : onSave) : undefined}
          >
            {nls.label_apply}
          </Button>
        </div>
      )}
    </>
  );
};

export default Header;
