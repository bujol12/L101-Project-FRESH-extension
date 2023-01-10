export CONFIG_FILE=Rationale_Analysis/training_config/classifiers/${CLASSIFIER:?"Set classifier"}.jsonnet
export CUDA_DEVICE=${CUDA_DEVICE:?"set cuda device"}

export TRAIN_DATA_PATH=${DATA_BASE_PATH:?"set data base path"}/train.jsonl
export DEV_DATA_PATH=$DATA_BASE_PATH/dev.jsonl
export TEST_DATA_PATH=$DATA_BASE_PATH/test.jsonl

export OUTPUT_BASE_PATH=${OUTPUT_DIR:-outputs}/${DATASET_NAME:?"Set dataset name"}/$CLASSIFIER/${EXP_NAME:?"Set Exp name"}

export SEED=${RANDOM_SEED:-100}

if [[ -f "${OUTPUT_BASE_PATH}/metrics.json" && -z "$again" ]]; then
    echo "${OUTPUT_BASE_PATH}/metrics.json exists ... . Not running Training ";
else 
    echo "${OUTPUT_BASE_PATH}/metrics.json does not exist ... . TRAINING ";
    allennlp train -s $OUTPUT_BASE_PATH --include-package Rationale_Analysis --force $CONFIG_FILE
fi


function predictions {
    if [[ -f "$1"  && -z "$again" ]]; then 
        echo "$1 exists .. Not Predicting";
    else 
        echo "$1 do not exist ... Predicting";
        allennlp predict \
        --output-file $1 \
        --batch-size ${BSIZE} \
        --use-dataset-reader \
        --dataset-reader-choice validation \
        --include-package Rationale_Analysis \
        --cuda-device ${CUDA_DEVICE:?"set cuda device"} \
        --silent \
        $OUTPUT_BASE_PATH/model.tar.gz $2;
    fi;
}

chmod +x $OUTPUT_BASE_PATH/model.tar.gz

predictions ${OUTPUT_BASE_PATH}/train_out.jsonl $TRAIN_DATA_PATH
predictions ${OUTPUT_BASE_PATH}/dev_out.jsonl $DEV_DATA_PATH
predictions ${OUTPUT_BASE_PATH}/test_out.jsonl $TEST_DATA_PATH