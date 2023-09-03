DATA_DIR = "../data/doi_10"
OUTPUT_DIR = "./output/scenario_1"

#run the training script
for subject in 1 4 5 7 11 12 15 16
do
    python train.py DATA.DIR $DATA_DIR OUTPUT.LOG_DIR $OUTPUT_DIR DATA.SUBJECT $subject
done

#run train_adv
