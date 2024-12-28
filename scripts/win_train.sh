#!/bin/bash

log_with_timestamp() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1"
}

# log_with_timestamp "Starting the training"
# nohup python train.py --train_data_dir data/person --output_dir results/exp_person  > logs/training.log 2>&1
# log_with_timestamp "Training Complete"
log_with_timestamp "Starting testing on Winobias dataset"
nohup python test.py --train_data_dir data/person --output_dir results/exp_person --evaluation_type winobias --num_test_samples 50 --template_key 0 --concept 'woman' 'man' --clip_attributes 'a woman' 'a man' > logs/testing.log 2>&1
log_with_timestamp "Testing Complete"