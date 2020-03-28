export const genWidgetWrapperStyle = (theme: any) => ({
  container: {
    flexGrow: 1,
    display: 'flex',
    flexDirection: 'column',
    maxWidth: '100%',
    maxHeight: '100%',
  },
  icon: {
    width: '20px',
    marginLeft: '8px',
    display: 'inline-block',
  },
  header: {
    position: 'relative',
    width: '100%',
    padding: theme.spacing(0.5),
    backgroundColor: '#fff',
    '& h3': {
      padding: theme.spacing(0.5, 3),
      overflow: 'hidden',
      textOverflow: 'ellipsis',
      whiteSpace: 'nowrap',
    },
  },
  actions: {
    zIndex: 100,
    position: 'absolute',
    top: theme.spacing(1),
    left: 0,
    width: '100%',
    display: 'flex',
    justifyContent: 'flex-end',
    alignItems: 'center',
    fontSize: '16px',
  },
  hidden: {
    visibility: 'hidden',
  },
  link: {
    cursor: 'pointer',
    padding: '0 2px',
  },
});
