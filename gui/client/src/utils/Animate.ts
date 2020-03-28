import { interpolate } from "d3";
export const default_duration = 250;

export function animate(param: any) {
  return {
    requestId: -1,
    duration: param.duration,
    curve: param.curve,
    currentV: param.curve[0],
    onAnimate: param.onAnimate || function() {},
    onEnd: param.onEnd || function() {},
    startTimeStamp: 0,
    ease:
      param.ease ||
      function(t: any, d: any) {
        return t / d;
      },
    step: function(timeStamp: any) {
      // timeStamp start from 0, timeStamp == window.performance.now()
      if (!this.startTimeStamp) this.startTimeStamp = timeStamp;

      var timePassed = timeStamp - this.startTimeStamp;

      this.currentV =
        this.curve[0] +
        (this.curve[1] - this.curve[0]) * this.ease(timePassed, this.duration);

      this.onAnimate(
        this.currentV > this.curve[1] ? this.curve[1] : this.currentV
      );
      // this.currentV < this.curve[1]
      if (this.duration > timePassed) {
        this.requestId = window.requestAnimationFrame(this.step.bind(this));
      } else {
        this.onEnd();
      }
    },
    start: function() {
      this.requestId = window.requestAnimationFrame(this.step.bind(this));
    },
    stop: function() {
      window.cancelAnimationFrame(this.requestId);
    }
  };
}

const getTransData = (
  originData: any,
  targetData: any,
  key: string,
  elapsed: number,
  duration: number
) => {
  const gap = targetData[key] - originData[key];
  return originData[key] + (gap * elapsed) / duration;
};

export const getStackedBarTransistData = (
  originDatas: any,
  targetDatas: any,
  elapsed: number = 0,
  duration: number
) => {
  elapsed = elapsed > duration ? duration : elapsed;
  return targetDatas.map((targetData: any) => {
    const originData = originDatas.find(
      (originData: any) =>
        originData.x === targetData.x && originData.color === targetData.color
    ) || {
      ...targetData,
      setting: {
        fill: targetData.setting.fill,
        x: 0,
        y: 0,
        width: 0,
        height: 0
      }
    };
    const { x, y, width, height } = originData.setting;
    const { setting } = targetData;
    const gapX = setting.x - x,
      gapY = setting.y - y,
      gapWidth = setting.width - width,
      gapHeight = setting.height - height;
    return {
      ...originData,
      setting: {
        ...originData.setting,
        x: x + (gapX * elapsed) / duration,
        y: y + (gapY * elapsed) / duration,
        width: width + (gapWidth * elapsed) / duration,
        height: height + (gapHeight * elapsed) / duration,
        fill: setting.fill
      }
    };
  });
};

export const getBubbleTransistData = (
  originDatas: any,
  targetDatas: any,
  elapsed: number = 0,
  duration: number
) => {
  elapsed = elapsed > duration ? duration : elapsed;
  return targetDatas.map((targetData: any) => {
    const originData = originDatas.find(
      (originData: any) => originData.text === targetData.text
    ) || { ...targetData, radius: 0 };
    const colorGetter = interpolate(originData.color, targetData.color);
    const res = {
      ...originData,
      xPos: getTransData(originData, targetData, "xPos", elapsed, duration),
      yPos: getTransData(originData, targetData, "yPos", elapsed, duration),
      textXPos: getTransData(
        originData,
        targetData,
        "textXPos",
        elapsed,
        duration
      ),
      textYPos: getTransData(
        originData,
        targetData,
        "textYPos",
        elapsed,
        duration
      ),
      radius: getTransData(originData, targetData, "radius", elapsed, duration),
      color: colorGetter(elapsed / duration)
    };
    return res;
  });
};
