export const pieLabelDecorator = (v: string, pie: any, radius: number, centroid: any): string => {
  const ln = String(v).length;
  const {startAngle, endAngle} = pie;
  const angle = endAngle - startAngle;
  let w: any;

  if (isNaN(angle) || angle * radius < 28) {
    return '...';
  }

  const adjacent = Math.abs(centroid[1]);

  if (angle >= Math.PI * 2) {
    w = adjacent;
  } else {
    const useAngle = centroid[0] * centroid[1] < 0 ? startAngle : endAngle;
    const refAngle = centroid[1] >= 0 ? Math.PI : centroid[0] < 0 ? Math.PI * 2 : 0;

    const tan = Math.tan(Math.abs(refAngle - useAngle));
    const opposite = tan * adjacent;
    const labelWidth =
      refAngle >= startAngle && refAngle < endAngle
        ? Math.abs(centroid[0]) + opposite
        : Math.abs(centroid[0]) - opposite;
    const maxLabelWidth = radius - 24;

    w = labelWidth > maxLabelWidth || labelWidth < 0 ? maxLabelWidth : labelWidth;
  }

  const APPROX_FONT_WIDTH = 10;

  if (adjacent < APPROX_FONT_WIDTH * 2) {
    // console.log(centroid, adjacent, APPROX_FONT_WIDTH * 2);
    // console.log("xx 2");
    // return '...';
  }

  if (ln * APPROX_FONT_WIDTH > w) {
    return String(v).slice(0, w / APPROX_FONT_WIDTH) + '…';
  }

  return v;
};

export const getFitFontSize = (v: string, width: number, height: number): number => {
  let fontSize: number = Math.min(width / 2, height);
  let size: any;
  const div = document.createElement('div');
  div.innerHTML = v;
  document.body.appendChild(div);
  const gap = Math.max(v.length, 5);
  do {
    fontSize = fontSize - gap;
    div.setAttribute(
      'style',
      `font-size: ${fontSize}px; visibility:hidden; position:absolute; width:auto; height:auto;`
    );
    size = div.getBoundingClientRect();
  } while (size.height > height * 0.95 || size.width > width * 0.9);
  document.body.removeChild(div);
  return fontSize;
};

export const sliceText = (v: string, len: number = 15) => {
  return v.length > len ? `${v.slice(0, len)}...` : v;
};
