model=$1
dataset=$2
PROJECT_PATH=../

DATA_PATH=${PROJECT_PATH}/datasets/${dataset}

case $model in
base)
    MODEL_PATH=${PROJECT_PATH}/models/wmt19.de-en/wmt19.de-en.ffn8192.pt
    ;;
finetune)
    MODEL_PATH=${PROJECT_PATH}/models/${dataset}_${model}/checkpoint_best.pt
    ;;
*)
    echo "error, `model` value not be implemented !"
    exit 1
esac

OUTPUT_PATH=${PROJECT_PATH}/outputs/${dataset}_${model}
mkdir -p $OUTPUT_PATH

CUDA_VISIBLE_DEVICES=0 python $PROJECT_PATH/fairseq_cli/generate.py $DATA_PATH\
    --gen-subset test \
    --path $MODEL_PATH \
    --beam 4 --lenpen 0.6 --max-len-a 1.2 --max-len-b 10 --source-lang de --target-lang en \
    --scoring sacrebleu --quiet \
    --batch-size 128 \
    --tokenizer moses --remove-bpe | tee $OUTPUT_PATH/generate.txt

# grep ^S "$OUTPUT_PATH"/generate.txt | cut -f2- > "$OUTPUT_PATH"/src
# grep ^T "$OUTPUT_PATH"/generate.txt | cut -f2- > "$OUTPUT_PATH"/ref
# grep ^H "$OUTPUT_PATH"/generate.txt | cut -f3- > "$OUTPUT_PATH"/hyp
# grep ^D "$OUTPUT_PATH"/generate.txt | cut -f3- > "$OUTPUT_PATH"/hyp.detok