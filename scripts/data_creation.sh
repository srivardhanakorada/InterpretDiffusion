#!/bin/bash

echo "Data Creation Started"
echo "============================================================"

echo "Start : Male"
nohup python data_creation.py --config_path env/dataset_config_male.json > logs/data_creation_male.log 2>&1 &
sleep 100

echo "Start : Female"
nohup python data_creation.py --config_path env/dataset_config_female.json > logs/data_creation_female.log 2>&1 &
sleep 100

echo "============================================================"
echo "Data Creation Stopped"