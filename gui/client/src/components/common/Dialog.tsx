import React, {FC, useContext} from 'react';
import Dialog from '@material-ui/core/Dialog';
import DialogContent from '@material-ui/core/DialogContent';
import DialogTitle from '@material-ui/core/DialogTitle';
import DialogActions from '@material-ui/core/DialogActions';
import Button from '@material-ui/core/Button';
import Typography from '@material-ui/core/Typography';
import {rootContext} from '../../contexts/RootContext';
import {I18nContext} from '../../contexts/I18nContext';
import {DIALOG_MODE} from '../../utils/Consts';

interface IInfoDialogProps {
  open: boolean;
  title: string;
  content: React.ReactElement;
  onConfirm: Function;
  onCancel: Function;
  confirmLabel: string;
  cancelLabel: string;
  mode: DIALOG_MODE;
}

const InfoDialog: FC<IInfoDialogProps> = props => {
  const {setDialog} = useContext(rootContext);
  const {nls} = useContext(I18nContext);
  const {
    open = false,
    title,
    content,
    onConfirm = () => {},
    onCancel = () => {},
    confirmLabel,
    cancelLabel,
    mode,
  } = props;
  const showCancelBtn = mode === DIALOG_MODE.CONFIRM;
  const handleDialogClose = () => {
    onConfirm();
    setDialog({open: false});
  };
  const handleDialogCancel = () => {
    onCancel();
    setDialog({open: false});
  };

  return (
    <Dialog
      onClose={handleDialogClose}
      transitionDuration={open ? 300 : 0}
      aria-labelledby="customized-dialog-title"
      open={open}
    >
      <DialogTitle id="customized-dialog-title">{title || nls.label_dialog_title}</DialogTitle>
      <DialogContent dividers>
        <Typography gutterBottom>{content || nls.label_dialog_content}</Typography>
      </DialogContent>
      <DialogActions>
        <Button onClick={handleDialogClose} color="default">
          {confirmLabel || nls.label_confirm}
        </Button>
        {showCancelBtn && (
          <Button onClick={handleDialogCancel} color="default">
            {cancelLabel || nls.label_cancel}
          </Button>
        )}
      </DialogActions>
    </Dialog>
  );
};

export default InfoDialog;
