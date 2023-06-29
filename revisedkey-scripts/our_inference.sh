dataset=$1
DATASTORE_PATH=$2
PROJECT_PATH=./

case $dataset in
koran)
    DSTORE_SIZE=524400
    temperature=100
    k=8
    lambda=0.8
    ;;
it)
    DSTORE_SIZE=3613350
    temperature=10
    k=8
    lambda=0.7
    ;;
law)
    DSTORE_SIZE=19070000
    temperature=10
    k=4
    lambda=0.8
    ;;
medical)
    DSTORE_SIZE=6903320
    temperature=10
    k=4
    lambda=0.8
    ;;
*)
    echo "error, `dataset` value not be implemented !"
    exit 1
esac

DATA_PATH=${PROJECT_PATH}/datasets/${dataset}
MODEL_PATH=${PROJECT_PATH}/models/wmt19.de-en/wmt19.de-en.ffn8192.pt
OUTPUT_PATH=${PROJECT_PATH}/outputs/knnmt_${dataset}_${model}
mkdir -p $OUTPUT_PATH

CUDA_VISIBLE_DEVICES=0 python $PROJECT_PATH/experimental_generate.py $DATA_PATH \
    --gen-subset test \
    --path $MODEL_PATH --arch transformer_wmt19_de_en_with_datastore \
    --beam 4 --lenpen 0.6 --max-len-a 1.2 --max-len-b 10 --source-lang de --target-lang en \
    --scoring sacrebleu --quiet \
    --batch-size 128 \
    --tokenizer moses --remove-bpe \
    --model-overrides "{'load_knn_datastore': True, 'use_knn_datastore': True,
    'dstore_filename': '$DATASTORE_PATH', 'dstore_size': $DSTORE_SIZE, 'dstore_fp16': True, 'k': $k, 'probe': 32,
    'knn_sim_func': 'do_not_recomp_l2', 'use_gpu_to_search': True, 'move_dstore_to_mem': True, 'no_load_keys': True,
    'knn_lambda_type': 'fix', 'knn_lambda_value': $lambda, 'knn_temperature_type': 'fix', 'knn_temperature_value': $temperature,
     }" \
    | tee $OUTPUT_PATH/generate.txt
