export const cloneObj = (obj: any): any => {
  return JSON.parse(JSON.stringify(obj));
};

export const id = (prefix: string = 'id') =>
  `${prefix}_${Math.random()
    .toString(36)
    .substr(2, 16)}`;

export const namespace = (prefix: string[] = [], name: string) => {
  return `${['infini', ...prefix].join('.')}:${name}`;
};

export const isValidValue = (val: any) => {
  return val !== null && val !== undefined;
};

export const formatSource = (source: string) => source.split('.')[1];
