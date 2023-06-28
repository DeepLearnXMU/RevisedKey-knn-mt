dataset=$1

num=4
ratio=0.7
threshold=2.0
ffn_size=8192
PROJECT_PATH=../

DIMENSION=1024
SOURCE_MODEL_PATH=${PROJECT_PATH}/models/wmt19.de-en/wmt19.de-en.ffn8192.pt
SOURCE_DATASTORE_PATH=${PROJECT_PATH}/datastores/${dataset}_base
TARGET_MODEL_PATH=${PROJECT_PATH}/models/${dataset}_finetune/checkpoint_best.pt
TARGET_DATASTORE_PATH=${PROJECT_PATH}/datastores/${dataset}_finetune
VOCAB_PATH=${PROJECT_PATH}/models/wmt19.de-en/fairseq-vocab.txt
SAVE_PATH=${PROJECT_PATH}/save/news_to_${dataset}

mkdir -p "$SAVE_PATH"

CUDA_VISIBLE_DEVICES=0 python $PROJECT_PATH/revisedkey/train.py \
    --dataset $dataset \
    --dimension $DIMENSION \
    --sample-num $num \
    --source-dstore-mmap $SOURCE_DATASTORE_PATH \
    --target-dstore-mmap $TARGET_DATASTORE_PATH \
    --source-model $SOURCE_MODEL_PATH \
    --target-model $TARGET_MODEL_PATH \
    --vocab-path $VOCAB_PATH \
    --learning-rate 5e-5 \
    --select-topk 32 \
    --filte-data \
    --filte-unreliable-ratio $ratio \
    --remove-top \
    --norm-weight $threshold \
    --ffn-size $ffn_size \
    --max-tokens 10000 --dstore-max-tokens 10000 \
    --max-epoch 100 --valid-interval-epoch 5 --log-interval-update 10 \
    --save-path $SAVE_PATH