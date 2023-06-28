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
    DATASTORE_PATH=${PROJECT_PATH}/datastores/${dataset}_${model}
    ;;
finetune)
    DATASTORE_PATH=${PROJECT_PATH}/datastores/${dataset}_${model}
    ;;
*)
    echo "error, `model` value not be implemented !"
    exit 1
esac

case $model in
base)
    DATASTORE_PATH=${PROJECT_PATH}/datastores/${dataset}_${model}
    ;;
finetune)
    DATASTORE_PATH=${PROJECT_PATH}/datastores/${dataset}_${model}
    ;;
*)
    echo "error, `model` value not be implemented !"
    exit 1
esac

DIMENSION=1024
INDEX=${DATASTORE_PATH}/knn_index
SAVE_PATH=${DATASTORE_PATH}/retrieve_results_start0

python save_retrieve_result.py \
    --dstore-mmap $DATASTORE_PATH \
    --dstore-size $DSTORE_SIZE \
    --index $INDEX \
    --k 32 \
    --dstore-fp16 \
    --dimension $DIMENSION \
    --save $SAVE_PATH \
    --start-point 0
    