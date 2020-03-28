import React, {FC, createContext, useContext, ReactNode, useState} from 'react';
import axios, {AxiosRequestConfig} from 'axios';
import * as URL from '../utils/Endpoints';
import {authContext} from './AuthContext';
import {I18nContext} from './I18nContext';
import {rootContext} from './RootContext';
import {namespace} from '../utils/Helpers';
import {DB_TYPE, Params, QueryType} from '../types';
import {isDashboardReady, getDashboardById} from '../utils/Dashboard';

const axiosInstance = axios.create();
export const queryContext = createContext<any>({});

const {Provider} = queryContext;
const QueryProvider: FC<{children: ReactNode}> = ({children}) => {
  const {setUnauthStatus, setAuthStatus} = useContext(authContext);
  const {widgetSettings, setDialog, setSnackbar, globalConfig} = useContext(rootContext);
  const {nls} = useContext(I18nContext);
  const getDB = () => {
    const db = window.localStorage.getItem(namespace(['query'], 'db'));
    return db ? JSON.parse(db) : false;
  };
  const [DB, _setDB] = useState<DB_TYPE | false>(getDB());
  const setDB = (db: DB_TYPE) => {
    window.localStorage.setItem(namespace(['query'], 'db'), JSON.stringify(db));
    _setDB(db);
  };
  // ref: https://stackoverflow.com/questions/41515732/hide-401-console-error-in-chrome-dev-tools-getting-401-on-fetch-call/42986081#42986081
  // in chrome, 401 can not be catched
  // so it will be shown on the console
  const errorParser = (e: any) => {
    const statusCode = e.response && e.response.data.statusCode;
    const API_SERVER_ERROR = !statusCode;
    let errContent = statusCode && e.response.data.message + ',' + nls.tip_refresh_page;

    // update errContent based on error type
    errContent = API_SERVER_ERROR ? nls.label_server_not_found : errContent;

    // reject content
    let err = {
      isError: true,
      timestamp: Date.now(),
      statusCode,
      errContent,
    };

    // 401, just reset auth status
    if (statusCode === 401) {
      setUnauthStatus();
    } else {
      // show dialog and return to config db page
      // since most error are db errors
      setDialog({
        open: true,
        content: errContent,
        onConfirm: () => {
          _setDB(false);
        },
      });
    }

    return Promise.reject(err);
  };

  const getAxiosConfig = (params: any = {}): AxiosRequestConfig => {
    let userAuth = JSON.parse(
      window.localStorage.getItem(namespace(['login'], 'userAuth')) || '{}'
    );
    return {
      headers: {Authorization: `Token ${userAuth && userAuth.token}`, ...params},
    };
  };

  const getConnId = () => {
    let userAuth = JSON.parse(
      window.localStorage.getItem(namespace(['login'], 'userAuth')) || '{}'
    );
    return userAuth && userAuth.connId;
  };

  const setConnId = (connId: string) => {
    let userAuth = JSON.parse(
      window.localStorage.getItem(namespace(['login'], 'userAuth')) || '{}'
    );
    userAuth.connId = connId;
    setAuthStatus(userAuth);
  };

  const login = async (params: any) => {
    let url = URL.POST_LOG_IN;
    return await axiosInstance
      .post(url, params, getAxiosConfig())
      .then((res: any) => res.data)
      .catch(errorParser);
  };

  const getData = (params: Params) => {
    let url = URL.Query;
    return axiosInstance
      .post(url, {id: DB && DB.id, query: params}, getAxiosConfig())
      .then((res: any) => {
        return res.data && res.data.data && res.data.data.result;
      })
      .catch(errorParser);
  };

  const getRowBySql = (sql: string) => {
    let url = URL.Query;
    let params = {type: 'sql', sql};
    return axiosInstance
      .post(url, {id: DB && DB.id, query: params}, getAxiosConfig())
      .then((res: any) => {
        return res && res.data.data[0] && res.data.data[0].result;
      })
      .catch(errorParser);
  };

  const numMinMaxValRequest = (colName: string, source: string) => {
    let sql = `SELECT MIN(${colName}) as min , MAX(${colName}) as max FROM ${source}`;
    return getData({type: QueryType.sql, sql})
      .then((res: any) => {
        return res[0];
      })
      .catch(errorParser);
  };
  const getDashBoard = async (id: number) => {
    let dashboard: any = getDashboardById(id);
    const url = URL.POST_TABLES_DETAIL;
    const sources = await getAvaliableTables();
    const options: any[] = await Promise.all(
      sources.map(async (source: string) => {
        const [id, table] = [DB && DB.id, source];
        let res = await axiosInstance.post(url, {id, table}, getAxiosConfig());
        return res && res.data.data;
      })
    );
    const totals = await Promise.all(
      sources.map(async (source: string) => {
        const params = {type: QueryType.sql, sql: `select count(*) from ${source}`};
        const res = await getData(params);
        const key = Object.keys(res[0])[0];
        return res[0][key];
      })
    );
    let sourceOptions: any = {};
    let sourceTotal: any = {};
    sources.forEach((source: string, i: number) => {
      sourceOptions[source] = options[i].sort((item1: any, item2: any) =>
        item1.col_name > item2.col_name ? 1 : -1
      );
      sourceTotal[source] = totals[i];
    });

    // final
    dashboard.sources = sources;
    dashboard.sourceOptions = sourceOptions;
    dashboard.totals = sourceTotal;
    return dashboard;
  };

  const getTxtDistinctVal = (sql: string) => {
    return getData({sql, type: QueryType.sql});
  };

  const getAvaliableTables = () => {
    let url = URL.GET_TABLE_LIST;
    return axiosInstance
      .post(url, {id: DB && DB.id}, getAxiosConfig())
      .then((res: any) => {
        return res && res.data.data;
      })
      .catch(errorParser);
  };

  const getDashboardList = () => {
    // console.log("get dashboard list, userId:", auth.userId);
    let dashboards: any = [];
    Object.entries(localStorage).forEach(([key, value]) => {
      if (key.indexOf('infini.dashboard') !== -1) {
        let dashboardObj = JSON.parse(value);
        // check if the config is ready, if not delete it from localStorage
        if (isDashboardReady(dashboardObj, widgetSettings)) {
          dashboards.push(dashboardObj);
        } else {
          window.localStorage.removeItem(key);
        }
      }
    });
    return Promise.resolve([...dashboards]);
  };

  const saveDashboard = (dashboardConfigs: string, id: number) => {
    window.localStorage.setItem(namespace(['dashboard'], String(id)), dashboardConfigs);
  };

  const removeDashboard = (id: string) => {
    window.localStorage.removeItem(namespace(['dashboard'], id));
    setSnackbar({open: true, message: nls.tip_remove_dashboard_success});
  };

  const getDBs = async () => {
    let url = URL.GET_DBS;
    return await axiosInstance.get(url, getAxiosConfig()).catch(errorParser);
  };

  // reverse geocoding to find the country,
  // so that we can make the map bound to that country
  const getMapBound = (lng: string, lat: string, table: string) => {
    let url = URL.Query;
    const sql = `SELECT ${lng}, ${lat} FROM ${table} WHERE ${lng} IS NOT NULL AND ${lat} IS NOT NULL LIMIT 1`;
    let params: Params = {type: QueryType.sql, sql};
    return axiosInstance
      .post(url, {id: DB && DB.id, query: params}, getAxiosConfig())
      .then((res: any) => {
        const lngLat = res && res.data && res.data.data.result;
        if (lngLat.length > 0) {
          const geocodingUrl = `https://api.mapbox.com/geocoding/v5/mapbox.places/${lngLat[0][lng]},${lngLat[0][lat]}.json?access_token=${globalConfig.MAPBOX_ACCESS_TOKEN}`;

          return axiosInstance.get(geocodingUrl).then((res: any) => {
            const features = res && res.data && res.data.features;
            const country = features[features.length - 1];
            return country && country.bbox;
          });
        }
        return Promise.resolve(true);
      });
  };

  const generalRequest = (sql: string) => {
    return getData({sql, type: QueryType.sql});
  };

  // check if the browser is firefox
  const isFirefox = /firefox/i.test(navigator.userAgent);

  return (
    <Provider
      value={{
        getConnId,
        setConnId,
        login,
        getData,
        getRowBySql,
        numMinMaxValRequest,
        getTxtDistinctVal,
        getDashBoard,
        getDashboardList,
        saveDashboard,
        removeDashboard,
        getAvaliableTables,
        DB,
        getDB,
        setDB,
        getDBs,
        getMapBound,
        isFirefox,
        generalRequest,
      }}
    >
      {children}
    </Provider>
  );
};
export default QueryProvider;
