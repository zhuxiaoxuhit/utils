# transfer docx to txt
# $1 : wav目录  $2: 保存目录

echo "starting ..."

find $1 -type f -name "*."wav | while read line; do
    base_name=$(basename $line .wav)
    echo $base_name
    sox $1/$base_name.wav -b 16 -c 1 $2/$base_name.wav
done

echo "finished"
