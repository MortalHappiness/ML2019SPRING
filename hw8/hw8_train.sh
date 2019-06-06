train=$1

mkdir hw3_models
mkdir checkpoint

for ((i=1; i<=8; ++i))
do
    wget "https://github.com/MortalHappiness/ml_hw8_teacher_models/releases/download/1.0/model"$i".h5"
    mv "model"$i".h5" "hw3_models/model"$i".h5"
done

python hw8_train.py $train
