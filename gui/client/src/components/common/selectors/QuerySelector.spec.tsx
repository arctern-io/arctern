import React from 'react';
import Selector, { useOptions } from './QuerySelector'
import cases from 'jest-in-case';
import { shallow, configure } from 'enzyme';
import Adapter from 'enzyme-adapter-react-16';
import { QueryCount } from '../../../utils/EditorHelper'
configure({ adapter: new Adapter() });

// test views
describe(`<Selector /> with common props`, () => {
  const initProps = {
    currOpt: { value: 'aaa' },
    placeholder: "click to select"
  }
  const selector = shallow(<Selector {...initProps} />)
  it(`should select 'aaa' at the moment`, () => {
    expect(selector.contains(<div>aaa</div>)).toBe(true);
  })
  it(`shoud have no input element`, () => {
    expect(selector.find('input').length).toBe(0);
  })
  it(`should change currOpt when prop change`, () => {
    selector.setProps({ ...initProps, currOpt: { value: 'bbb' } })
    expect(selector.contains(<div>bbb</div>)).toBe(true);
  })
})
describe(`<Selector /> with no currOpt`, () => {
  const initProps = {
    currOpt: {},
    placeholder: "click to select"
  }
  const selector = shallow(<Selector {...initProps} />)
  it(`should be in add status when no currOpt`, () => {
    expect(selector.childAt(0).childAt(0).text()).toEqual('click to select')
  });
})

// test hooks
const options = [
  { label: 'aaa', value: 'aaa' },
  { label: 'bbb', value: 'bbb' },
  { label: 'ccc', value: 'ccc' },
  { label: 'ddd', value: 'ddd' },
];
const Container = ({ children, query }: any) => {
  return children(useOptions(query));
}
function setUp(query: Function) {
  const returnVal: any = {};
  shallow(<Container query={query}>{
    (val: any) => {
      Object.assign(returnVal, val);
      return null
    }
  }</Container>)
  return returnVal;
}

cases('useOptions', async ({ query, first_options, filter_text }: any) => {
  const res = setUp(query);
  expect(res.filter_text).toBe(undefined);
  expect(res.visible_options).toEqual([])
  // test query
  await res.showMore(res.filter_text, 0);
  expect(res.visible_options).toEqual([first_options[0]])
  // test filter
  const e = { target: { value: filter_text } }
  res.changeFilter(e);
  setTimeout(() => {
    expect(res.visible_options).toEqual(first_options[0])
    res.clearFilter();
    expect(res.visible_options).toEqual(first_options)
  }, 500)

}, {
  basic: {
    query: async (filter_text: string | undefined) => filter_text ? options : [options[0]],
    first_options: options,
    filter_text: 'aa'
  },
})