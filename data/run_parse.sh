# train 
python3 scripts/parse_esnli.py \
    -train1 csv/esnli_train_1.csv \
    -train2 csv/esnli_train_2.csv \
    -dev csv/esnli_dev.csv \
    -test csv/esnli_test.csv \
    --split train \
    --split dev \
    --split test \
    -output_dir parsed/

