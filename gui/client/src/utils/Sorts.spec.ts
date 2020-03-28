import {typeSortGetter} from './Sorts';

test('Sorts', () => {
  const testSort = typeSortGetter('date' as any, 'x');
  const dateArr = [{x: '2019-01-01'}, {x: '2000-01-01'}];

  expect(dateArr.sort(testSort)).toEqual([{x: '2000-01-01'}, {x: '2019-01-01'}]);

  const defaultSort = typeSortGetter('number' as any, 'x');
  const numberArr = [{x: 3}, {x: 1}, {x: 4}];
  expect(numberArr.sort(defaultSort)).toEqual([{x: 1}, {x: 3}, {x: 4}]);

  const textSort = typeSortGetter('text' as any, 'y');
  const textArr = [{y: 'c'}, {y: 'a'}, {y: 'b'}];
  expect(textArr.sort(textSort)).toEqual([{y: 'a'}, {y: 'b'}, {y: 'c'}]);
});
