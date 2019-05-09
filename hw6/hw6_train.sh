train_x=$1
train_y=$2
test_x=$3
dict_txt=$4

python embedding.py $1 $3 $4
python hw6_train.py $1 $2 $4