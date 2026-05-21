#!/bin/bash

# Define default parameters
TRAINED_MODEL_PATH="output/trained_models/t5-small-epoch10-test-train"
DATA_PATH="data/nl_bltl/pcc_bltl.json"
OUTPUT_DIR="output/evaluated_results"
#INPUT_TEXT="The value of x should always be greater than 5"

# Run the inference script with parameters
python inference_t5.py \
  --model_path ${TRAINED_MODEL_PATH} \
  --data_path ${DATA_PATH} \
  --output_dir ${OUTPUT_DIR}
echo "Evaluation completed. Check the output above for results."