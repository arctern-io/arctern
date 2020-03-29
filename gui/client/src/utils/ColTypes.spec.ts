import {
  isNumCol,
  isFloatCol,
  isIntCol,
  isDateCol,
  isTextCol,
  getColType,
  getNumType,
} from './ColTypes';
import {NumberTypes, FloatTypes, IntTypes, DateTypes, TextTypes} from '../config';

test('isNumCol', () => {
  const isNum = isNumCol(NumberTypes[0]);
  const isNum2 = isNumCol(FloatTypes[0]);
  const isNum3 = isNumCol(IntTypes[0]);
  const isNum4 = isNumCol(DateTypes[0]);
  const isNum5 = isNumCol(TextTypes[0]);

  expect(isNum).toEqual(true);
  expect(isNum2).toEqual(true);
  expect(isNum3).toEqual(true);
  expect(isNum4).toEqual(false);
  expect(isNum5).toEqual(false);
});

test('isFloat', () => {
  const isFloat = isFloatCol(NumberTypes[0]);
  const isFloat2 = isFloatCol(FloatTypes[0]);
  const isFloat3 = isFloatCol(IntTypes[0]);
  const isFloat4 = isFloatCol(DateTypes[0]);
  const isFloat5 = isFloatCol(TextTypes[0]);

  expect(isFloat).toEqual(false);
  expect(isFloat2).toEqual(true);
  expect(isFloat3).toEqual(false);
  expect(isFloat4).toEqual(false);
  expect(isFloat5).toEqual(false);
});

test('isInt', () => {
  const isInt = isIntCol(NumberTypes[0]);
  const isInt2 = isIntCol(FloatTypes[0]);
  const isInt3 = isIntCol(IntTypes[0]);
  const isInt4 = isIntCol(DateTypes[0]);
  const isInt5 = isIntCol(TextTypes[0]);

  expect(isInt).toEqual(true);
  expect(isInt2).toEqual(false);
  expect(isInt3).toEqual(true);
  expect(isInt4).toEqual(false);
  expect(isInt5).toEqual(false);
});

test('isDate', () => {
  const isDate = isDateCol(NumberTypes[0]);
  const isDate2 = isDateCol(FloatTypes[0]);
  const isDate3 = isDateCol(IntTypes[0]);
  const isDate4 = isDateCol(DateTypes[0]);
  const isDate5 = isDateCol(TextTypes[0]);

  expect(isDate).toEqual(false);
  expect(isDate2).toEqual(false);
  expect(isDate3).toEqual(false);
  expect(isDate4).toEqual(true);
  expect(isDate5).toEqual(false);
});

test('isText', () => {
  const isText = isTextCol(NumberTypes[0]);
  const isText2 = isTextCol(FloatTypes[0]);
  const isText3 = isTextCol(IntTypes[0]);
  const isText4 = isTextCol(DateTypes[0]);
  const isText5 = isTextCol(TextTypes[0]);

  expect(isText).toEqual(false);
  expect(isText2).toEqual(false);
  expect(isText3).toEqual(false);
  expect(isText4).toEqual(false);
  expect(isText5).toEqual(true);
});

test('getColType', () => {
  const colType1 = getColType(NumberTypes[0]);
  const colType2 = getColType(FloatTypes[0]);
  const colType3 = getColType(IntTypes[0]);
  const colType4 = getColType(DateTypes[0]);
  const colType5 = getColType(TextTypes[0]);

  expect(colType1).toEqual('number');
  expect(colType2).toEqual('number');
  expect(colType3).toEqual('number');
  expect(colType4).toEqual('date');
  expect(colType5).toEqual('text');
});

test('getNumType', () => {
  const colType1 = getNumType(NumberTypes[0]);
  const colType2 = getNumType(FloatTypes[0]);
  const colType3 = getNumType(TextTypes[0]);

  expect(colType1).toEqual('int');
  expect(colType2).toEqual('float');
  expect(colType3).toEqual('unknown');
});
