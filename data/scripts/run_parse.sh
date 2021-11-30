# train 
python3 parse_esnli.py \
    -train1 ../csv/esnli_train_1.csv \
    -train2 ../csv/esnli_train_2.csv \
    -split train \
    -output_dir ../parsed

# dev 
python3 parse_esnli.py \
    -dev ../csv/esnli_dev.csv \
    -split dev \
    -output_dir ../parsed

# test 
python3 parse_esnli.py \
    -test ../csv/esnli_test.csv \
    -split test \
    -output_dir ../parsed
