input_file=$1
output_file=$2

python process_test_data.py $input_file ./hw1_best_model/x_test.npy
python hw1_best_test.py $output_file