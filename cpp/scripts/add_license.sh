FOLDER=$1
if [[ -z ${FOLDER} ]]; then
    echo usage $0 [folder_to_add_license]
    exit
else
    echo good
fi

FILES=`find ${FOLDER} | grep -E "(*\.cpp$|*\.h$|*\.cu$)"`
echo formating ${FILES} ...
for f in ${FILES}; do
  if (grep Copyright $f);then 
    echo "No need to copy the License Header to $f"
  else
    cat license.txt $f > $f.new
    mv $f.new $f
    echo "License Header copied to $f"
  fi 
done   
