# transfer docx to txt
# $1 : docx目录  $2: 保存目录

echo "starting ..."
find $1 -type f -name "*."docx | while read line; do
    base_name=$(basename $line .docx)
    echo $base_name
    docx2txt $1/$base_name.docx > $2/$base_name.txt
done
echo "finished"
