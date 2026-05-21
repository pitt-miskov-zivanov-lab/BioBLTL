#!/bin/bash

# Define default parameters
MODEL="t5-small"
DATA_DIR="data/nl_bltl"
DATA_BASENAME="synbltl_100_s1"
OUTPUT_DIR="output/trained_models"
BATCH_SIZE=8
NUM_EPOCHS=1
LEARNING_RATE=1e-4
#TRAINED_MODEL_PATH="output/trained_models/t5-large-epoch2-test-train"    

# Run the training script with parameters
python main.py \
  --model ${MODEL} \
  --data_dir ${DATA_DIR} \
  --output_dir ${OUTPUT_DIR} \
  --data_basename ${DATA_BASENAME} \
  --batch_size ${BATCH_SIZE} \
  --num_epochs ${NUM_EPOCHS} \
  --learning_rate ${LEARNING_RATE} \
  # --trained_model_path ${TRAINED_MODEL_PATH}

echo "Training completed. Check ${OUTPUT_DIR} for results."