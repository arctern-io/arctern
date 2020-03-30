import React, {FC, useState} from 'react';
import {color} from '../../utils/Colors';
import {sliceText} from '../../widgets/Utils/Decorators'
const Legend: FC<any> = props => {
  const {legendData = []} = props;
  const [expand, setExpand] = useState<any>('');
  const onClick = () => {
    setExpand(expand === '' ? 'expand' : '');
  };

  if (legendData && legendData.length) {
    return (
      <div className={`legend ${expand}`}>
        <div className="title" onClick={onClick}>
          {legendData[0].colName}
        </div>
        <div className="content">
          <ul>
            {legendData.map((d: any) => {
              return (
                <li key={d.as}>
                  <span className="mark" style={{background: `${d.color || color(d.as)}`}} />
                  {sliceText(d.label)}
                </li>
              );
            })}
          </ul>
        </div>
      </div>
    );
  }

  return <></>;
};

export default Legend;
