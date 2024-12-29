#!/bin/bash

echo "============================================================"
echo "Starting the training & testing sequences"
echo "============================================================"

echo "============================================================"
echo "Male concepts started"
echo "============================================================"

echo "Training for Male Concept 0 Started"
nohup python train.py --train_data_dir data/male/concept_0 --output_dir results/exp_male/concept_0 --device "cuda:0" --num_train_epochs 40 > logs/train_male_concept_0.log 2>&1
wait
echo "Training for Male Concept 0 Completed"
sleep 1
echo "Infering for Male Concept 0 Started"
nohup python test.py --train_data_dir data/male/concept_0 --output_dir results/exp_male/concept_0 --num_test_samples 100 --prompt "a doctor" --device "cuda:0" > logs/test_male_concept_0.log 2>&1
wait
echo "Infering for Male Concept 0 Completed"
sleep 1

echo "Training for Male Concept 1 Started"
nohup python train.py --train_data_dir data/male/concept_1 --output_dir results/exp_male/concept_1 --device "cuda:0" --num_train_epochs 40 > logs/train_male_concept_1.log 2>&1
wait
echo "Training for Male Concept 1 Completed"
sleep 1
echo "Infering for Male Concept 1 Started"
nohup python test.py --train_data_dir data/male/concept_1 --output_dir results/exp_male/concept_1 --num_test_samples 100 --prompt "a doctor" --device "cuda:0" > logs/test_male_concept_1.log 2>&1
wait
echo "Infering for Male Concept 1 Completed"
sleep 1

echo "Training for Male Concept 2 Started"
nohup python train.py --train_data_dir data/male/concept_2 --output_dir results/exp_male/concept_2 --device "cuda:0" --num_train_epochs 40 > logs/train_male_concept_2.log 2>&1
wait
echo "Training for Male Concept 2 Completed"
sleep 1
echo "Infering for Male Concept 2 Started"
nohup python test.py --train_data_dir data/male/concept_2 --output_dir results/exp_male/concept_2 --num_test_samples 100 --prompt "a doctor" --device "cuda:0" > logs/test_male_concept_2.log 2>&1
wait
echo "Infering for Male Concept 2 Completed"
sleep 1

echo "============================================================"
echo "Male concepts finished"
echo "============================================================"


echo "============================================================"
echo "female concepts started"
echo "============================================================"

echo "Training for female Concept 0 Started"
nohup python train.py --train_data_dir data/female/concept_0 --output_dir results/exp_female/concept_0 --device "cuda:0" --num_train_epochs 40 > logs/train_female_concept_0.log 2>&1
wait
echo "Training for female Concept 0 Completed"
sleep 1
echo "Infering for female Concept 0 Started"
nohup python test.py --train_data_dir data/female/concept_0 --output_dir results/exp_female/concept_0 --num_test_samples 100 --prompt "a doctor" --device "cuda:0" > logs/test_female_concept_0.log 2>&1
wait
echo "Infering for female Concept 0 Completed"
sleep 1

echo "Training for female Concept 1 Started"
nohup python train.py --train_data_dir data/female/concept_1 --output_dir results/exp_female/concept_1 --device "cuda:0" --num_train_epochs 40 > logs/train_female_concept_1.log 2>&1
wait
echo "Training for female Concept 1 Completed"
sleep 1
echo "Infering for female Concept 1 Started"
nohup python test.py --train_data_dir data/female/concept_1 --output_dir results/exp_female/concept_1 --num_test_samples 100 --prompt "a doctor" --device "cuda:0" > logs/test_female_concept_1.log 2>&1
wait
echo "Infering for female Concept 1 Completed"
sleep 1

echo "Training for female Concept 2 Started"
nohup python train.py --train_data_dir data/female/concept_2 --output_dir results/exp_female/concept_2 --device "cuda:0" --num_train_epochs 40 > logs/train_female_concept_2.log 2>&1
wait
echo "Training for female Concept 2 Completed"
sleep 1
echo "Infering for female Concept 2 Started"
nohup python test.py --train_data_dir data/female/concept_2 --output_dir results/exp_female/concept_2 --num_test_samples 100 --prompt "a doctor" --device "cuda:0" > logs/test_female_concept_2.log 2>&1
wait
echo "Infering for female Concept 2 Completed"
sleep 1

echo "============================================================"
echo "female concepts finished"
echo "============================================================"