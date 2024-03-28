

# $1 : input目录  $2: ouput目录

echo "starting ..."

find $1 -type f -name "*."txt | while read line;
do
    base_name=$(basename $line .txt)
    sed '/^\s*$/d' $1/$base_name.txt > $2/$base_name.txt
done

echo "finished"
