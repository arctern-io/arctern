import {COLUMN_TYPE} from './Consts';
import {NumberTypes, FloatTypes, IntTypes, DateTypes, TextTypes} from '../config';

export const isNumCol = (type: string = ''): boolean => {
  return NumberTypes.some((item: any) => item === type.toUpperCase());
};

export const isFloatCol = (type: string = ''): boolean => {
  return FloatTypes.some((item: any) => item === type.toUpperCase());
};

export const isIntCol = (type: string = ''): boolean => {
  return IntTypes.some((item: any) => item === type.toUpperCase());
};

export const isDateCol = (type: string = ''): boolean => {
  return DateTypes.some((item: any) => item === type.toUpperCase());
};

export const isTextCol = (type: string = ''): boolean => {
  return TextTypes.some((item: any) => item === type.toUpperCase());
};

export const getColType = (type: string): COLUMN_TYPE => {
  if (isNumCol(type)) {
    return COLUMN_TYPE.NUMBER;
  }
  if (isDateCol(type)) {
    return COLUMN_TYPE.DATE;
  }
  if (isTextCol(type)) {
    return COLUMN_TYPE.TEXT;
  }
  return COLUMN_TYPE.UNKNOWN;
};

export const getNumType = (type: string): string => {
  if (isFloatCol(type)) {
    return 'float';
  }

  if (isIntCol(type)) {
    return 'int';
  }

  return 'unknown';
};
