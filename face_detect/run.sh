echo "dir $1"

# sh run.sh ../crawler/kanako_image out_kanako

for file in `\find $1 -maxdepth 1 -type f`; do
  cmd="python detect.py $file $2"
  echo "$cmd"
done
