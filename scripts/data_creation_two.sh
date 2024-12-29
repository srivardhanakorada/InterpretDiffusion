#!/bin/bash

echo "Data Creation Started"
echo "============================================================"

echo "Start : Car"
nohup python data_creation.py --config_path env/dataset_config_car.json > logs/data_creation_car.log 2>&1 &
sleep 100

echo "Start : Chair"
nohup python data_creation.py --config_path env/dataset_config_table.json > logs/data_creation_table.log 2>&1 &
sleep 100

echo "============================================================"
echo "Data Creation Stopped"