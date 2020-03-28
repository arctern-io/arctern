import {COLUMN_TYPE} from './Consts';

export const typeSortGetter = (type: COLUMN_TYPE, attr: string): any => {
  switch (type) {
    case COLUMN_TYPE.DATE:
      return (a: any, b: any) => new Date(a[attr]).getTime() - new Date(b[attr]).getTime();
    case COLUMN_TYPE.TEXT:
      return (a: any, b: any) => ('' + a[attr]).localeCompare(b[attr]);
    case COLUMN_TYPE.NUMBER:
    default:
      return (a: any, b: any) => a[attr] - b[attr];
  }
};
