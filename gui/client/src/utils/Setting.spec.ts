import {makeSetting} from './Setting';

test('makeSetting', () => {
  const configHandler: any = () => 1;
  const onAfterSqlCreate = () => 2;
  const config: any = {};

  const setting = makeSetting({
    type: 's',
    dimensions: [],
    measures: [],
    icon: 'aDPnSXXERGQlkOFadQKIxfk5tXw',
    configHandler: configHandler,
    onAfterSqlCreate: onAfterSqlCreate,
  });

  expect(setting.type).toBe('s');
  expect(setting.enable).toBe(true);
  expect(setting.isServerRender).toBe(false);
  expect(setting.configHandler(config)).toBe(1);
  expect(setting.onAfterSqlCreate()).toBe(2);
});
