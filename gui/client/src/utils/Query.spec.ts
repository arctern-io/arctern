import {DataQuery, getLinkData} from './Query';

test('DataQuery', async () => {
  const requester: any = jest.fn(() => {
    return Promise.resolve({result: 'res'});
  });
  const cacheExpiresIn = 3000;
  const onResponse: any = jest.fn();
  const onRequest: any = jest.fn();
  const qParams: any = [{sql: 1}, {sql: 2}];

  const testQuery = new DataQuery({
    requester,
    cacheExpiresIn,
    onResponse,
    onRequest,
  });

  const res = await testQuery.q(qParams);
  expect(onResponse.mock.calls.length).toBe(2);
  expect(onRequest.mock.calls.length).toBe(2);
  expect(res).toStrictEqual(['res', 'res']);
  expect(requester.mock.calls.length).toBe(2);

  const res2 = await testQuery.q(qParams);
  expect(onResponse.mock.calls.length).toBe(2);
  expect(onRequest.mock.calls.length).toBe(4);
  expect(requester.mock.calls.length).toBe(2);
  expect(res2).toStrictEqual(['res', 'res']);
});

test('getLinkData', () => {
  const config: any = {
    linkId: 'linkId',
  };

  const config2: any = {
    linkId: 'linkId2',
  };

  const response: any = {linkId: 1};
  expect(getLinkData(response, config)).toBe(1);
  expect(getLinkData(response, config2)).toStrictEqual([]);
});
