#!/bin/bash
# Run 5-fold CV for DAVIS and KIBA

# Using the MixingDTA environment python
PYTHON_BIN="/home/lwfvx/miniforge3/envs/MixingDTA/bin/python"

run_cv() {
    DATASET_NAME=$1
    DATA_DIR_NAME=$2
    
    echo "Running 5-fold CV for $DATASET_NAME..."
    
    for fold in {1..5}; do
        echo "  Fold $fold..."
        # Data paths relative to project root
        TEST_FILE="data/Structure_independent/${DATA_DIR_NAME}/test_${fold}.csv"
        OUTPUT_DIR="outputs/mixingdta/${DATASET_NAME}/fold_${fold}"
        
        # Override dataset.path, model.output_dir, model.fold
        # We use ++ for extra safety to ensure it adds/overrides
        $PYTHON_BIN run_benchmark.py \
            model=mixingdta \
            dataset=${DATASET_NAME} \
            ++dataset.path=${TEST_FILE} \
            ++model.output_dir=${OUTPUT_DIR} \
            ++model.fold=${fold} \
            ++model.task=warm_start
            
        if [ $? -ne 0 ]; then
            echo "Error running fold $fold for $DATASET_NAME"
            # Optional: exit on error?
            # exit 1
        fi
    done
}

# Run DAVIS
run_cv "sequence_davis" "DAVIS"

# Run KIBA
run_cv "sequence_kiba" "KIBA"
