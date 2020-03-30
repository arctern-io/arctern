import React from 'react';
import { shallow, configure } from 'enzyme';
import { useStatus } from './Helper'
import cases from 'jest-in-case'
import Adapter from 'enzyme-adapter-react-16';
configure({ adapter: new Adapter() });

const Container = ({ children, status }: any) => {
  return children(useStatus(status))
}

function setUp(initStatus: string) {
  const returnVal: any = {};
  shallow(
    <Container status={initStatus}>
      {(val: any) => {
        Object.assign(returnVal, val);
        return null;
      }}
    </Container>)
  return returnVal;
}

cases('useSelectorStatus', ({ initStatus, addStatus, seletingStatus }: any) => {
  const res = setUp(initStatus);
  expect(res.status).toBe(initStatus)
  res.determineStatus('');
  expect(res.status).toBe(addStatus)
  res.setSelectingStatus();
  expect(res.status).toBe(seletingStatus)
}, {
  basic: {
    initStatus: 'selected',
    addStatus: 'add',
    seletingStatus: 'selectColumn'
  },
  random: {
    initStatus: 'add',
    addStatus: 'add',
    seletingStatus: 'selectColumn'
  }
})