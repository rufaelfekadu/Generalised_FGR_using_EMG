DATA_DIR="../../data/doi_10"
OUTPUT_DIR="../output/scenario_2"
subjects=(1 4 5 7 11 12 15 16)

python ../train.py \
    DATA.PATH $DATA_DIR \
    OUTPUT.LOG_DIR $OUTPUT_DIR'/cnn' \
    DATA.SUBJECT $subjects \
    TRAIN.BATCH_SIZE 64 \
    TRAIN.NUM_EPOCHS 100 \
    TRAIN.LR 0.001 

