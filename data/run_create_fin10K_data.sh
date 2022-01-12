FIN10K=/tmp2/yshuang/fintext/codes.fin.contextualized/passage_alignment/type2-sentences
mkdir fin10k

for FILEPATH in ${FIN10K}/*;do
    filecode=${FILEPATH##*/}

    echo "Createing data for bert..."
    pattern=.txt
    replace=.jsonl
    echo "File output: ${filecode/$pattern/$replace}"
    python3 scripts/create_fin10k_data.py \
        -input_path $FILEPATH \
        -output_path fin10k/${filecode/$pattern/$replace} \
        -model_type bert

    echo "Createing data for t5..."
    pattern=.txt
    replace=.tsv
    echo "File output: ${filecode/$pattern/$replace}"
    python3 scripts/create_fin10k_data.py \
        -input_path $FILEPATH \
        -output_path fin10k/${filecode/$pattern/$replace} \
        -model_type t5
done

