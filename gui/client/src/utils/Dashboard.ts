import {isReadyToRender} from './EditorHelper';
import {WidgetConfig, Dashboard} from '../types';
import {namespace} from './Helpers';

export const isDashboardReady = (dashboardConfig: any, widgetSettings: any): boolean => {
  const hasId = typeof dashboardConfig.id !== 'undefined';
  const hasTitle = typeof dashboardConfig.title !== 'undefined';
  const hasUserId = typeof dashboardConfig.userId !== 'undefined';
  const hasCorrectConfigType = Array.isArray(dashboardConfig.configs);
  const hasSources = Array.isArray(dashboardConfig.sources) && dashboardConfig.sources.length > 0;

  if (!hasId || !hasTitle || !hasUserId) {
    // console.log(`!hasId || !hasTitle || !hasUserId`);
    return false;
  }

  if (!hasCorrectConfigType || !hasSources) {
    // console.log(`!hasCorrectConfigType || !hasSources`);
    return false;
  }

  // check config valid
  const isConfigValid =
    dashboardConfig.configs.length === 0 ||
    dashboardConfig.configs.some((config: WidgetConfig) => {
      // check config valid
      const {sourceReady, dimensionsReady, measuresReady} = isReadyToRender(
        config,
        widgetSettings[config.type]
      );

      return sourceReady.isReady && dimensionsReady.isReady && measuresReady.isReady;
    });

  if (!isConfigValid) {
    // console.log('isConfigValid');
    return false;
  }

  return true;
};

const genNewDashboard = (id: number = 0): Dashboard => ({
  id: id,
  demo: false,
  title: `Dashboard-${id}`,
  userId: 'infini',
  configs: [],
  createdAt: new Date().toUTCString(),
  modifyAt: new Date().toUTCString(),
  sources: [],
  sourceOptions: {},
});

export const getDashboardById = (id: number) => {
  const local = window.localStorage.getItem(namespace(['dashboard'], String(id)));

  if (local) {
    return JSON.parse(local);
  }

  return genNewDashboard(id);
};
