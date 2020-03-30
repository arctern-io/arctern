import React, {FC, useState, useContext} from 'react';
import {color} from '../../utils/Colors';
import {I18nContext} from '../../contexts/I18nContext';

const Legend: FC<any> = props => {
  const {nls} = useContext(I18nContext);
  const {legendData} = props;
  const [expand, setExpand] = useState<any>('expand');

  const onClick = () => {
    setExpand(expand === '' ? 'expand' : '');
  };

  if (legendData.showLegend) {
    return (
      <div className={`legend ${expand}`}>
        <div className="title" onClick={onClick}>
          {nls.label_legend}
        </div>
        <div className="content">
          <ul>
            {legendData.data.map((d: any) => {
              return (
                <li key={d.as}>
                  <span className="mark" style={{background: `${d.color || color(d.as)}`}} />
                  {d.legendLabel}
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
