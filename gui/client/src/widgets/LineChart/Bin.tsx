import React, {FC} from 'react';

interface IBinProps {
  binData: any;
}

const Bin: FC<IBinProps> = props => {
  const {binData} = props;
  binData.showBin = false;
  if (binData.showBin) {
    return (
      <div className="bin">
        <div className="title">Bin: {binData.bin}</div>
      </div>
    );
  }

  return <></>;
};

export default Bin;
