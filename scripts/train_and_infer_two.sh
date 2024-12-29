#!/bin/bash

echo "============================================================"
echo "Starting the training & testing sequences"
echo "============================================================"

echo "============================================================"
echo "Car concepts started"
echo "============================================================"

echo "Training for Car Concept 0 Started"
nohup python train.py --train_data_dir data/car/concept_0 --output_dir results/exp_car/concept_0 --device "cuda:1" --num_train_epochs 40 > logs/train_car_concept_0.log 2>&1
wait
echo "Training for Car Concept 0 Completed"
sleep 1
echo "Infering for Car Concept 0 Started"
nohup python test.py --train_data_dir data/car/concept_0 --output_dir results/exp_car/concept_0 --num_test_samples 100 --prompt "a vehicle" --device "cuda:1" > logs/test_car_concept_0.log 2>&1
wait
echo "Infering for Car Concept 0 Completed"
sleep 1

echo "Training for Car Concept 1 Started"
nohup python train.py --train_data_dir data/car/concept_1 --output_dir results/exp_car/concept_1 --device "cuda:1" --num_train_epochs 40 > logs/train_car_concept_1.log 2>&1
wait
echo "Training for Car Concept 1 Completed"
sleep 1
echo "Infering for Car Concept 1 Started"
nohup python test.py --train_data_dir data/car/concept_1 --output_dir results/exp_car/concept_1 --num_test_samples 100 --prompt "a vehicle" --device "cuda:1" > logs/test_car_concept_1.log 2>&1
wait
echo "Infering for Car Concept 1 Completed"
sleep 1

echo "Training for Car Concept 2 Started"
nohup python train.py --train_data_dir data/car/concept_2 --output_dir results/exp_car/concept_2 --device "cuda:1" --num_train_epochs 40 > logs/train_car_concept_2.log 2>&1
wait
echo "Training for Car Concept 2 Completed"
sleep 1
echo "Infering for Car Concept 2 Started"
nohup python test.py --train_data_dir data/car/concept_2 --output_dir results/exp_car/concept_2 --num_test_samples 100 --prompt "a vehicle" --device "cuda:1" > logs/test_car_concept_2.log 2>&1
wait
echo "Infering for Car Concept 2 Completed"
sleep 1

echo "============================================================"
echo "Car concepts finished"
echo "============================================================"


echo "============================================================"
echo "Table concepts started"
echo "============================================================"

echo "Training for table Concept 0 Started"
nohup python train.py --train_data_dir data/table/concept_0 --output_dir results/exp_table/concept_0 --device "cuda:1" --num_train_epochs 40 > logs/train_table_concept_0.log 2>&1
wait
echo "Training for table Concept 0 Completed"
sleep 1
echo "Infering for table Concept 0 Started"
nohup python test.py --train_data_dir data/table/concept_0 --output_dir results/exp_table/concept_0 --num_test_samples 100 --prompt "a piece of furniture" --device "cuda:1" > logs/test_table_concept_0.log 2>&1
wait
echo "Infering for table Concept 0 Completed"
sleep 1

echo "Training for table Concept 1 Started"
nohup python train.py --train_data_dir data/table/concept_1 --output_dir results/exp_table/concept_1 --device "cuda:1" --num_train_epochs 40 > logs/train_table_concept_1.log 2>&1
wait
echo "Training for table Concept 1 Completed"
sleep 1
echo "Infering for table Concept 1 Started"
nohup python test.py --train_data_dir data/table/concept_1 --output_dir results/exp_table/concept_1 --num_test_samples 100 --prompt "a piece of furniture" --device "cuda:1" > logs/test_table_concept_1.log 2>&1
wait
echo "Infering for table Concept 1 Completed"
sleep 1

echo "Training for table Concept 2 Started"
nohup python train.py --train_data_dir data/table/concept_2 --output_dir results/exp_table/concept_2 --device "cuda:1" --num_train_epochs 40 > logs/train_table_concept_2.log 2>&1
wait
echo "Training for table Concept 2 Completed"
sleep 1
echo "Infering for table Concept 2 Started"
nohup python test.py --train_data_dir data/table/concept_2 --output_dir results/exp_table/concept_2 --num_test_samples 100 --prompt "a piece of furniture" --device "cuda:1" > logs/test_table_concept_2.log 2>&1
wait
echo "Infering for table Concept 2 Completed"
sleep 1

echo "============================================================"
echo "Table concepts finished"
echo "============================================================"