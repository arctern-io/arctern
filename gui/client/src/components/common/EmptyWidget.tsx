import React, {useContext} from 'react';
import {I18nContext} from '../../contexts/I18nContext';
import {MODE} from '../../utils/Consts';

const size = '500px';
const EmptyWidget = (props: any) => {
  const {nls} = useContext(I18nContext);
  const {setMode} = props;
  return (
    <div
      style={{
        width: size,
        height: size,
        border: 'dashed',
        borderWidth: '1px',
        display: 'flex',
        margin: '20px',
        alignItems: 'center',
        justifyContent: 'center',
        cursor: 'pointer',
      }}
      onClick={() => {
        setMode({
          mode: MODE.ADD,
          id: '',
        });
      }}
    >
      <h1>{`+ ${nls.label_add_new_widget}`}</h1>
    </div>
  );
};

export default EmptyWidget;
