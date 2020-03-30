import {WidgetConfig} from '../types';
import {cloneObj} from './Helpers';

export const exportCsv = (config: WidgetConfig, data: any) => {
  const filename = `infini-export-${config.source}-${config.title}.csv`;
  const cols = [
    ...config.dimensions.map(d => ({as: d.as, value: d.value})),
    ...config.measures.map(m => ({as: m.as, value: m.value})),
  ];
  const colsHeader = cols.map((c: any) => c.value);
  let csvContent = `data:text/csv;charset=utf-8,${colsHeader.join(',')}\r\n`;

  data.forEach((row: any) => {
    let newRow: any = [];
    cols.forEach((col: any) => {
      newRow.push(row[col.as]);
    });
    csvContent += newRow.join(',') + '\r\n';
  });
  const encodedUri = encodeURI(csvContent);
  // download
  const link = document.createElement('a');
  link.setAttribute('href', encodedUri);
  link.setAttribute('download', filename);
  document.body.appendChild(link); // Required for FF

  link.click(); // This will download the data file named "my_data.csv".
  document.body.removeChild(link);
  return {encodedUri, filename};
};

export const exportJson = (dashboardObj: any) => {
  const target = cloneObj(dashboardObj);
  delete target.sourceOptions;
  const filename = `infini-dashboard-${target.title}.json`;
  const csvContent = `data:text/json;charset=utf-8,${encodeURIComponent(
    JSON.stringify(target)
  )}\r\n`;

  // download
  const link = document.createElement('a');
  link.setAttribute('href', csvContent);
  link.setAttribute('download', filename);
  document.body.appendChild(link); // Required for FF
  link.click(); // This will download the data file named "my_data.csv".
  document.body.removeChild(link);
  return {filename, csvContent};
};
