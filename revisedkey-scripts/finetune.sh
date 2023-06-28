dataset=$1
PROJECT_PATH=../

DATA_PATH=${PROJECT_PATH}/datasets/$dataset
BASE_MODEL_PATH=${PROJECT_PATH}/models/wmt19.de-en/wmt19.de-en.ffn8192.pt
MODEL_SAVE_PATH=${PROJECT_PATH}/models/${dataset}_finetune

mkdir -p $MODEL_SAVE_PATH
cp $BASE_MODEL_PATH $MODEL_SAVE_PATH/checkpoint_last.pt

CUDA_VISIBLE_DEVICES=0 python ${PROJECT_PATH}/fairseq_cli/train.py \
    $DATA_PATH \
    --arch transformer_wmt19_de_en --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.1 \
    --lr 7e-4 \
    --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.2 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 4, "lenpen": 0.6, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --save-dir "$MODEL_SAVE_PATH" --keep-interval-updates 5 \
    --log-interval 1000 \
    --validate-interval-updates 1000 --save-interval-updates 2000 \
    --patience 30 --max-epoch 500
