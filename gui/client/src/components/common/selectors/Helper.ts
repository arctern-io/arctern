import { useState } from 'react';
export function useStatus(initStatus: string = "selected") {
  const [status, setStatus] = useState(initStatus);
  const determineStatus = (val: string) => {
    setStatus(val ? 'selected' : 'add')
  }
  const setSelectingStatus = () => {
    setStatus('selectColumn')
  }
  return { status, determineStatus, setSelectingStatus }
}