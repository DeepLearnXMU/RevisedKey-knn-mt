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

case $model in
base)
    TRAIN_DATASTORE_PATH=${PROJECT_PATH}/datastores/${dataset}_${model}
    ;;
finetune)
    TRAIN_DATASTORE_PATH=${PROJECT_PATH}/datastores/${dataset}_${model}
    ;;
*)
    echo "error, `model` value not be implemented !"
    exit 1
esac

DIMENSION=1024
INDEX=${TRAIN_DATASTORE_PATH}/knn_index
DATASTORE_PATH=${TRAIN_DATASTORE_PATH}/valid

python save_valid_retrieve_result.py \
    --dstore-mmap $DATASTORE_PATH \
    --dstore-size $DSTORE_SIZE \
    --index $INDEX \
    --k 32 \
    --dstore-fp16 \
    --dimension $DIMENSION \
    --save $DATASTORE_PATH \
    --start-point 0
    