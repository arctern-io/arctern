import {InfiniNode, InfiniCollection} from '.';

test('Infini', () => {
  const reducer = (d: any, acc: any) => {
    // console.log(d);
    return (acc || 0) + 1;
  };
  const s = () => '1';
  const infini = new InfiniCollection({reducer, stringify: s});

  const r1 = infini.create('table1');
  infini.appendChild(r1); // add extra one
  expect(r1 instanceof InfiniNode).toBe(true);
  expect(r1.source).toBe('table1');
  expect(r1.type).toBe('root');
  expect(r1.transform).toStrictEqual([]);
  expect(r1.children).toStrictEqual([]);
  expect(infini.children).toStrictEqual([r1, r1]);

  const n1Data = {
    transform: [{pivot: 'symbol', value: 'price', groupby: ['date']}],
  };
  const n1 = r1.add(n1Data);
  expect(n1.transform).toStrictEqual([{pivot: 'symbol', value: 'price', groupby: ['date']}]);
  expect(n1.source).toBe(r1);
  expect(n1.type).toBe('node');
  expect(r1.children).toStrictEqual([n1]);
  expect(r1.children.length).toBe(1);

  const n2Data = {
    transform: [{pivot: 'symbol2', value: 'price2', groupby: ['date2']}],
  };

  const n2 = r1.add(n2Data);
  expect(n2.transform).toStrictEqual([{pivot: 'symbol2', value: 'price2', groupby: ['date2']}]);
  expect(n2.source).toBe(r1);
  expect(n2.type).toBe('node');
  expect(r1.children).toStrictEqual([n1, n2]);
  expect(r1.children.length).toBe(2);

  const n3 = r1.add(n2);
  expect(n3.type).toBe('node');
  expect(n3.source).toBe(r1);

  expect(n3.transform).toStrictEqual([{pivot: 'symbol2', value: 'price2', groupby: ['date2']}]);
  expect(r1.children).toStrictEqual([n1, n2, n2]);

  const tr = n3.reduce();
  const tr2 = n3.reduceToString();
  expect(tr).toStrictEqual(2);
  expect(tr2).toStrictEqual('1');
});
