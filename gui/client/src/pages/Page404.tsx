import React, {useContext} from 'react';
import {I18nContext} from '../contexts/I18nContext';

const Page404: React.FC = () => {
  const {nls} = useContext(I18nContext);
  return <div className="404">{nls.label_404}</div>;
};

export default Page404;
