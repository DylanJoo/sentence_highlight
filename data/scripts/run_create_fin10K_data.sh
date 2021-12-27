# Bert's
for FILEPATH in /tmp2/jhju/data/*;do
    filecode=${FILEPATH##*/}
    pattern=.txt
    replace=.jsonl
    echo "Createing ${filecode/$pattern/$replace} ..."
    python3 create_fin10k_for_bert.py \
        -input_path /tmp2/jhju/data/${filecode} \
        -output_path ../fin10k/${filecode/$pattern/$replace}
done

# T5's
for FILEPATH in /tmp2/jhju/data/*;do
    filecode=${FILEPATH##*/}
    pattern=.txt
    replace=.tsv
    echo "Createing ${filecode/$pattern/$replace} ..."
    python3 create_fin10k_for_t5.py \
        -input_path /tmp2/jhju/data/${filecode} \
        -output_path ../fin10k/${filecode/$pattern/$replace}
done
