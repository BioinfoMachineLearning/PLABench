#!/bin/bash
# Run MixingDTA Experiments (Warm, Cold-Drug, Cold-Target) for DAVIS and KIBA

# Using the MixingDTA environment python
PYTHON_BIN="/home/lwfvx/miniforge3/envs/MixingDTA/bin/python"

# Function to run 5-fold CV
run_cv() {
    DATASET_CONFIG=$1 # e.g. davis_warm, davis_cold_drug
    DATA_DIR_NAME=$2  # DAVIS or KIBA
    TASK=$3           # warm_start, cold_drug, cold_target
    
    echo "========================================================"
    echo "Running 5-fold CV for $DATASET_CONFIG ($TASK)"
    echo "========================================================"
    
    for fold in {1..5}; do
        echo "  Fold $fold..."
        
        # Determine Input File Path based on Task
        if [ "$TASK" == "warm_start" ]; then
            TEST_FILE="data/Structure_independent/${DATA_DIR_NAME}/test_${fold}.csv"
        elif [ "$TASK" == "cold_drug" ]; then
            # Cold Drug typically has one test set for all folds? 
            # DTA_DataBase/cold/DAVIS/test_{fold}.pkl does NOT exist.
            # DTA_DataBase/cold/DAVIS/test_Drug.pkl exists.
            # Using placeholder CSV path, assuming we point to a single file
            # But the fold argument is still needed for loading the correct model weight.
            TEST_FILE="data/Structure_independent/${DATA_DIR_NAME}/cold/test_drug.csv"
        elif [ "$TASK" == "cold_target" ]; then
             TEST_FILE="data/Structure_independent/${DATA_DIR_NAME}/cold/test_target.csv"
        fi
        
        OUTPUT_DIR="outputs/mixingdta/${DATASET_CONFIG}/fold_${fold}"
        
        # Override dataset.path, model.output_dir, model.fold, model.task
        $PYTHON_BIN run_benchmark.py \
            model=mixingdta \
            dataset=${DATASET_CONFIG} \
            ++dataset.path=${TEST_FILE} \
            ++model.output_dir=${OUTPUT_DIR} \
            ++model.fold=${fold} \
            ++model.task=${TASK} \
            ++model.device=cuda
            
        if [ $? -ne 0 ]; then
            echo "Error running fold $fold for $DATASET_CONFIG"
        fi
    done
}

# --- DAVIS ---
# 1. Warm Start
run_cv "davis_warm" "DAVIS" "warm_start"

# 2. Cold Drug
run_cv "davis_cold_drug" "DAVIS" "cold_drug"

# 3. Cold Target
run_cv "davis_cold_target" "DAVIS" "cold_target"

# --- KIBA ---
# 4. Warm Start
run_cv "kiba_warm" "KIBA" "warm_start"

# 5. Cold Drug
run_cv "kiba_cold_drug" "KIBA" "cold_drug"

# 6. Cold Target
run_cv "kiba_cold_target" "KIBA" "cold_target"
