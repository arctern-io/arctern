import React, {FC, useState, useRef, useEffect} from 'react';
import {renderToString} from 'react-dom/server';
import * as globalConfig from '../config';
import {WidgetSettings} from '../types';
import {namespace} from '../utils/Helpers';

// interface
interface IRootContext {
  theme: any;
  saveTheme: Function;
  themes: string[];
  themeMap: any;
  dialog: any;
  setDialog: any;
  snakebar: any;
  setSnackbar: any;
  showTooltip: any;
  hideTooltip: any;
  globalConfig: any;
  widgetSettings: WidgetSettings;
}

interface IShowTooltip {
  position: any;
  tooltipData: any;
  titleGetter: Function;
  contentGetter: Function;
  isShowTitle?: boolean;
}

// load all settings from folder
const widgetSettings: any = {};
const themeMap: any = {};
const themes: string[] = [];
function importAllWidgets(r: any) {
  r.keys().forEach((key: string) => {
    const m = r(key);
    const defaultM = m.default;
    if (defaultM && defaultM.enable) {
      widgetSettings[defaultM.type] = defaultM;
    }
  });
}
importAllWidgets(require.context('../widgets', true, /settings\.ts$/));
function importThemes(r: any) {
  r.keys().forEach((key: string) => {
    const m = r(key);
    const defaultM = m.default;
    if (m.default) {
      const _key = key.split('Theme')[0].substring(2);
      themes.push(_key);
      themeMap[_key] = defaultM;
    }
  });
}
importThemes(require.context('../themes', false, /Theme\.ts$/));

export const rootContext = React.createContext<IRootContext>({
  theme: {},
  themes: [],
  themeMap: {},
  saveTheme: () => {},
  dialog: {},
  setDialog: () => {},
  snakebar: {},
  setSnackbar: () => {},
  showTooltip: () => {},
  hideTooltip: () => {},
  globalConfig: {},
  widgetSettings: widgetSettings,
});
const {Provider} = rootContext;

// put global singletons here: dialog, tooltip...
const RootProvider: FC<{children: React.ReactNode}> = ({children}) => {
  // color theme
  const auth = window.localStorage.getItem(namespace(['login'], 'userAuth'));
  const currTheme = (auth && JSON.parse(auth).theme) || themes[0];
  const [theme, setTheme] = useState<any>(currTheme);
  // dialog state
  const [dialog, setDialog] = useState<any>({});
  // snackbar state
  const [snakebar, setSnackbar] = useState<any>({});
  // internal tooltip ref
  const _tooltipTitle = useRef<HTMLDivElement>(null);
  const _tooltipContent = useRef<HTMLDivElement>(null);
  const _tooltipWrapper = useRef<HTMLDivElement>(null);
  const _timeout: any = useRef(null);

  const saveTheme = (theme: string) => {
    const auth = window.localStorage.getItem(namespace(['login'], 'userAuth'));
    if (auth) {
      window.localStorage.setItem(
        namespace(['login'], 'userAuth'),
        JSON.stringify({...JSON.parse(auth), theme})
      );
    }
    setTheme(theme);
  };

  // tooltip is kind special component in React World
  // most time we need to keep updating the tooltip on mouse moving
  // but we don't want react to keep rerending the whole app
  // so we update tooltip's position and content by traditional way
  // inside component, just call showTooltip/hideTooltip to show/hide the tooltip
  const showTooltip = (props: IShowTooltip) => {
    if (_timeout.current) {
      clearTimeout(_timeout.current);
    }
    const {position, tooltipData, titleGetter, contentGetter, isShowTitle = true} = props;
    if (!position) {
      return;
    }

    if (_tooltipTitle.current) {
      _tooltipTitle.current.innerHTML = isShowTitle ? renderToString(titleGetter(tooltipData)) : '';
    }

    if (_tooltipContent.current) {
      _tooltipContent.current.innerHTML = renderToString(contentGetter(tooltipData));
      // ReactDOM.render(contentGetter(tooltipData), _tooltipContent.current)
    }

    if (_tooltipWrapper.current) {
      let size = _tooltipWrapper.current.getBoundingClientRect();
      let top = position.event.pageY - size.height - 20;
      if (position.event.clientY < size.height + 20) {
        top = position.event.pageY + 50;
      }

      let left = position.event.pageX - size.width / 2;
      if (position.event.pageX + size.width / 2 > document.body.clientWidth) {
        left = position.event.pageX - size.width;
      }
      if (position.event.pageX - size.width / 2 < 0) {
        left = 10;
      }
      _tooltipWrapper.current.style.left = left + 'px';
      _tooltipWrapper.current.style.top = top + 'px';
    }
  };

  const hideTooltip = (e: any, timeout: number = 40): void => {
    if (_timeout.current) {
      clearTimeout(_timeout.current);
    }
    _timeout.current = setTimeout(() => {
      if (_tooltipWrapper.current) {
        _tooltipWrapper.current.style.left = '-999999999px';
      }
    }, timeout);
  };
  useEffect(() => {
    if (_tooltipWrapper.current) {
      _tooltipWrapper.current.style.left = `-999999px`;
    }
  }, []);

  return (
    <Provider
      value={{
        theme,
        themes,
        themeMap,
        saveTheme,
        dialog,
        setDialog,
        snakebar,
        setSnackbar,
        showTooltip,
        hideTooltip,
        globalConfig,
        widgetSettings,
      }}
    >
      {children}
      <div
        className="tooltip"
        ref={_tooltipWrapper}
        onMouseEnter={() => {
          if (_timeout.current) {
            clearTimeout(_timeout.current);
          }
        }}
        onMouseLeave={e => {
          hideTooltip(e, 0);
        }}
      >
        <div className="title" ref={_tooltipTitle}></div>
        <div className="content" ref={_tooltipContent}></div>
      </div>
    </Provider>
  );
};

export default RootProvider;
