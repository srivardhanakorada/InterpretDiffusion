#!/bin/bash

echo "Starting the training and testing sequence for all concepts..."
echo "============================================================"

# Training and Testing Commands for Concept 0
echo "Training for Concept 0 started..."
nohup python train.py --train_data_dir data/female/concept_0 --output_dir results/exp_female/concept_0 > logs/train_concept_0.log 2>&1
wait  # Wait for the command to finish
sleep 5
echo "Training for Concept 0 completed. Logs available at logs/train_concept_0.log"

echo "Testing for Concept 0 started..."
nohup python test.py --train_data_dir data/female/concept_0 --output_dir results/exp_female/concept_0 --num_test_samples 10 --prompt "a doctor" > logs/test_concept_0.log 2>&1
wait  # Wait for the command to finish
sleep 5
echo "Testing for Concept 0 completed. Logs available at logs/test_concept_0.log"

# Training and Testing Commands for Concept 1
echo "Training for Concept 1 started..."
nohup python train.py --train_data_dir data/female/concept_1 --output_dir results/exp_female/concept_1 > logs/train_concept_1.log 2>&1
wait  # Wait for the command to finish
sleep 5
echo "Training for Concept 1 completed. Logs available at logs/train_concept_1.log"

echo "Testing for Concept 1 started..."
nohup python test.py --train_data_dir data/female/concept_1 --output_dir results/exp_female/concept_1 --num_test_samples 10 --prompt "a doctor" > logs/test_concept_1.log 2>&1
wait  # Wait for the command to finish
sleep 5
echo "Testing for Concept 1 completed. Logs available at logs/test_concept_1.log"

# Training and Testing Commands for Concept 2
echo "Training for Concept 2 started..."
nohup python train.py --train_data_dir data/female/concept_2 --output_dir results/exp_female/concept_2 > logs/train_concept_2.log 2>&1
wait  # Wait for the command to finish
sleep 5
echo "Training for Concept 2 completed. Logs available at logs/train_concept_2.log"

echo "Testing for Concept 2 started..."
nohup python test.py --train_data_dir data/female/concept_2 --output_dir results/exp_female/concept_2 --num_test_samples 10 --prompt "a doctor" > logs/test_concept_2.log 2>&1
wait  # Wait for the command to finish
sleep 5
echo "Testing for Concept 2 completed. Logs available at logs/test_concept_2.log"

echo "============================================================"
echo "Training and testing sequence for all concepts completed successfully."
echo "Check the logs directory for detailed logs of each operation."