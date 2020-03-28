import React, {FC, useState, useEffect} from 'react';
import './Spinner.scss';

interface ISpinner {
  delay?: number;
}

const Spinner: FC<ISpinner> = (props: any = {}) => {
  const {delay = 100} = props;
  const [loading, setLoading] = useState<string>('');

  useEffect(() => {
    let timeout = setTimeout(() => {
      setLoading('spinner');
    }, delay);

    return () => {
      clearTimeout(timeout);
    };
  });
  return <div className={loading}></div>;
};

export default Spinner;
