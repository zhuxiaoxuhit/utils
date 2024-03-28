# $1 : mp3目录  $2: 保存目录

echo "starting ..."

find $1 -type f -name "*."mp3 | while read line; 
do
    base_name=$(basename $line .mp3)
    ffmpeg -i $1/$base_name.mp3 $2/$base_name.wav
done


#find $1 -type f -name "*."wav | while read line; do
#    base_name=$(basename $line .wav)
#    sox $1/$base_name.wav -b 16 -c 1 $2/$base_name.wav
#    echo $base_name
#done

echo "finished"
