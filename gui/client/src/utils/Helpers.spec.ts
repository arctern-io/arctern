import {isValidValue, namespace, id, cloneObj} from './Helpers';

test('cloneObj', () => {
  let a = {};
  let b = cloneObj(a);
  expect(b).toEqual(a);

  let c = {a: 2, b: 3, d: {e: 4}};
  let d = cloneObj(c);
  expect(d).toEqual(c);
});

test('id', () => {
  let myId = id();
  expect(typeof myId).toBe('string');
  expect(myId.split('_')[0]).toEqual('id');
  expect(myId).toEqual(expect.stringContaining('id'));

  let myId2 = id('xx');
  expect(typeof myId2).toBe('string');
  expect(myId2.split('_')[0]).toEqual('xx');
  expect(myId2).toEqual(expect.stringContaining('xx'));
});

test('namespace', () => {
  let myNamespace = namespace(['myprefix'], 'testname');
  expect(typeof myNamespace).toBe('string');
  expect(myNamespace).toBe(`infini.myprefix:testname`);

  let myNamespace2 = namespace(['myprefix1', 'myprefix2'], 'testname');
  expect(myNamespace2).toBe(`infini.myprefix1.myprefix2:testname`);
});

test('isValidValue', () => {
  const invalid1 = null;
  const invalid2 = undefined;
  const valid5 = '';
  const valid6 = false;
  const valid1 = 'string';
  const valid2: any = [];
  const valid3 = {};
  const valid4 = Symbol;

  expect(isValidValue(invalid1)).toBe(false);
  expect(isValidValue(invalid2)).toBe(false);
  expect(isValidValue(valid1)).toBe(true);
  expect(isValidValue(valid2)).toBe(true);
  expect(isValidValue(valid3)).toBe(true);
  expect(isValidValue(valid4)).toBe(true);
  expect(isValidValue(valid5)).toBe(true);
  expect(isValidValue(valid6)).toBe(true);
});
