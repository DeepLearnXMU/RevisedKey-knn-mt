model=$1
dataset=$2
PROJECT_PATH=../

case $dataset in
koran)
    DSTORE_SIZE=524400
    ;;
it)
    DSTORE_SIZE=3613350
    ;;
law)
    DSTORE_SIZE=19070000
    ;;
medical)
    DSTORE_SIZE=6903320
    ;;
*)
    echo "error, `dataset` value not be implemented !"
    exit 1
esac

DIMENSION=1024
DATA_PATH=${PROJECT_PATH}/datasets/${dataset}

case $model in
base)
    MODEL_PATH=${PROJECT_PATH}/models/wmt19.de-en/wmt19.de-en.ffn8192.pt
    TRAIN_DATASTORE_PATH=${PROJECT_PATH}/datastores/${dataset}_${model}
    ;;
finetune)
    MODEL_PATH=${PROJECT_PATH}/models/${dataset}_${model}/checkpoint_best.pt
    TRAIN_DATASTORE_PATH=${PROJECT_PATH}/datastores/${dataset}_${model}
    ;;
*)
    echo "error, `model` value not be implemented !"
    exit 1
esac

DATASTORE_PATH=${TRAIN_DATASTORE_PATH}/valid
mkdir -p $DATASTORE_PATH

CUDA_VISIBLE_DEVICES=0 python save_valid_kv.py $DATA_PATH \
    --dataset-impl mmap \
    --task translation \
    --valid-subset valid \
    --path $MODEL_PATH \
    --max-tokens 16384 --skip-invalid-size-inputs-valid-test \
    --decoder-embed-dim $DIMENSION --dstore-fp16 --dstore-size $DSTORE_SIZE --dstore-mmap $DATASTORE_PATH