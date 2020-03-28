import React from 'react';
import Selector, { useOptions, filterOptions } from './SimpleSelector'
import { shallow, configure, mount } from 'enzyme';
import Adapter from 'enzyme-adapter-react-16';
import { QueryCount } from '../../../utils/EditorHelper'
import cases from 'jest-in-case';
configure({ adapter: new Adapter() });

const options = [
  { label: 'aaa', value: 'aaa' },
  { label: 'bbb', value: 'bbb' },
  { label: 'ccc', value: 'ccc' },
  { label: 'ddd', value: 'ddd' },
];
const bigOpts = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o'].map((item: string) => {
  return {
    label: item.repeat(3),
    value: item.repeat(3)
  }
})
// test views
describe(`<Selector /> with common props`, () => {
  const initProps = {
    currOpt: options[0],
    options: options,
    isShowCurrOpt: false,
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
    selector.setProps({ ...initProps, currOpt: options[1] })
    expect(selector.contains(<div>bbb</div>)).toBe(true);
  })
})

describe(`<Selector /> with no currOpt`, () => {
  it(`should be in add status when no currOpt`, () => {
    const initProps = {
      currOpt: {},
      options: options,
      isShowCurrOpt: false,
      placeholder: "click to select"
    }
    const selector = mount(<Selector {...initProps} />)
    expect(selector.childAt(0).childAt(0).text()).toEqual('click to select')
  });
})

describe(`<Selector /> in "selectColumn" status `, () => {
  const initProps = {
    currOpt: options[0],
    options: options,
    isShowCurrOpt: false,
  }
  const selector = mount(<Selector {...initProps} />)
  const button = selector.find('div').at(2);

  button.simulate('click');
  const input = selector.find('input')
  it(`shoud have only one input for filter options, 3 valible options and 1 unvalidated option`, () => {
    expect(input.length).toBe(1);
    expect(selector.find('ul')).toHaveLength(1)
    expect(selector.find('ul').children()).toHaveLength(4)
  })
  it(`shoud filter valid options when input's value change`, () => {
    input.simulate('change', {
      target: {
        value: 'b',
      },
    });
    expect(selector.find('ul').children()).toHaveLength(2)
  })
})

describe(`<Selector /> in "selectColumn" status with huge options`, () => {
  const initProps = {
    currOpt: bigOpts[0],
    options: bigOpts,
    isShowCurrOpt: false,
  }
  const selector = mount(<Selector {...initProps} />)
  const button = selector.find('div').at(2);
  button.simulate('click');
  it(`should show only ${QueryCount} options when all options's length largger than ${QueryCount}, and show more option, total ${QueryCount + 1} options`, () => {
    const options = selector.find('ul')
    expect(options.children()).toHaveLength(QueryCount + 1);
  })
  //TODO: how to test state changed by other component 
  it(`should add ${QueryCount} more options to show, if the rest options's length less than ${QueryCount}, show <NoMore />`, () => {
    // selector.instance().setShowOpts(2 * QueryCount)
    // const options = selector.find('ul')
    // expect(options.children()).toHaveLength(bigOpts.length);
  })
})

// test hooks
const Container = ({ children, valid_options }: any) => {
  return children(useOptions(valid_options));
}

function setUp(init_options: any[]) {
  const returnVal: any = {};
  shallow(<Container valid_options={init_options}>{
    (val: any) => {
      Object.assign(returnVal, val);
      return null
    }
  }</Container>)
  return returnVal;
}

cases('useOptions', ({ init_options, filter_text }: any) => {
  const res = setUp(init_options);
  expect(res.filter_text).toBe(undefined);
  expect(res.filtered_options).toEqual(init_options)
  expect(res.visible_options).toEqual(init_options.slice(0, QueryCount));

  res.showMore();
  expect(res.visible_options).toEqual(init_options.slice(0, QueryCount * 2))
  const e = { target: { value: filter_text } }
  res.changeFilter(e);
  // TODO: return new res before effect works, so use timeout to make sure all effects has been runned. Change it if better method found.
  setTimeout(() => {
    expect(res.filtered_options).toEqual(filterOptions(filter_text, init_options))
    res.clearFilter();
    expect(res.filtered_options).toEqual(init_options)
  }, 500)

}, {
  basic: {
    init_options: bigOpts,
    filter_text: 'aa'
  },
})