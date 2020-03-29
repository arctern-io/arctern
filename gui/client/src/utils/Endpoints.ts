declare global {
  interface Window {
    _env_: any;
  }
}
let endpoint = `http://127.0.0.1:5555`;
// let endpoint = `http://localhost:9000`;
// if (window._env_ && window._env_.API_URL) {
//   endpoint = window._env_.API_URL;
// }
export const Query = `${endpoint}/db/query`;
export const POST_TABLES_DETAIL = `${endpoint}/db/table/info`;
export const GET_TABLE_LIST = `${endpoint}/db/tables`;
export const GET_DBS = `${endpoint}/dbs`;
export const POST_DB_CONFIG = `${endpoint}/config/db`;
export const POST_LOG_IN = `${endpoint}/login`;
export const POST_LOG_OUT = `${endpoint}/logout`;
