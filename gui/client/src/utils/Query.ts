import {WidgetConfig, Query, Params, Data, DataCache} from '../types';
import {cloneObj} from './Helpers';

// requester
type Requester = (query: Params) => Promise<Data>;

// local types
type Cache = Map<
  string,
  {
    expire: number;
    data: Data;
  }
>;

type onResponse = (query: Query, data: Data) => void;
type onRequest = (query: Query) => void;

interface IQuery {
  // query cache, used for store a map of last query,
  // the key is sql, value is sql result
  cache: Cache;
  // cache will be expires in millionseconds
  cacheExpiresIn: number;
  // request exec, it accetps query multiple Query as params
  // and it will return a promise when all querys are resolved
  q: (querys: Query[]) => Promise<Data>;
  // fired on each query
  onRequest: onRequest;
  // fire on each query return
  onResponse: onResponse;
  // requester, used for send query
  requester: Requester;
}

type QueryParams = {
  requester: Requester;
  onResponse: onResponse;
  onRequest: onRequest;
  cacheExpiresIn?: number;
};

export class DataQuery implements IQuery {
  cache: Cache = new Map();
  requester: Requester;
  cacheExpiresIn = 0;
  onResponse: onResponse = () => {};
  onRequest: onRequest = () => {};
  constructor({requester, onResponse, onRequest, cacheExpiresIn = 1000 * 60 * 5}: QueryParams) {
    this.requester = requester;
    this.onResponse = onResponse;
    this.onRequest = onRequest;
    this.cacheExpiresIn = cacheExpiresIn;
  }
  q(querys: Query[]) {
    console.info('querys', querys);
    let promiseGrp: Promise<any>[] = [];
    let i = 0;
    let timestamp = Date.now();
    while (i < querys.length) {
      const query: Query = querys[i];
      const cacheKey = JSON.stringify(query);
      const cacheData = this.cache.get(cacheKey);
      const meta = cloneObj(query);
      meta.timestamp = timestamp;
      // before request
      this.onRequest(meta);
      // if we have cache and cache is not expire
      if (cacheData) {
        if (Date.now() < cacheData.expire) {
          setTimeout(() => {
            this.onResponse(query, cacheData.data);
          });
          promiseGrp.push(Promise.resolve(cacheData.data));
          i++;
          continue;
        }
      }

      // delete cache
      this.cache.delete(cacheKey);
      // send the request
      const queryPromise = this.requester(query.params).then((data: any) => {
        this.cache.set(cacheKey, {
          expire: Date.now() + this.cacheExpiresIn,
          data,
        });
        this.onResponse(meta, data);
        return data;
      });
      promiseGrp.push(queryPromise);
      i++;
    }

    return Promise.all(promiseGrp);
  }
}

export const getLinkData = (response: DataCache, config: WidgetConfig) => {
  let res: Data = [];
  if (config.linkId && response[config.linkId]) {
    res = response[config.linkId];
  }

  return res;
};
