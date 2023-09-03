DATA_DIR = "../data/doi_10"
OUTPUT_DIR = "./output/scenario_1"

#run the training script
for subject in 1 4 5 7 11 12 15 16
do
    python ../train.py --data_dir $DATA_DIR --output_dir "$OUTPUT_DIR/cnn" --epochs 100 \
    --batch_size 32 --lr 0.0001 --n_workers 4 --seed 42 --subject $subject --session 1 \
    --pos 1 
done

#run train_adv
