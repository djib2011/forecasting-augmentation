#!/bin/bash

python datasets/generate_from_forecaster.py --model-horizon 6 --real-insample 8
python datasets/generate_from_forecaster.py --model-horizon 6 --real-insample 10
python datasets/generate_from_forecaster.py --model-horizon 6 --real-insample 12
python datasets/generate_from_forecaster.py --model-horizon 6 --real-insample 14
python datasets/generate_from_forecaster.py --model-horizon 6 --real-insample 16
python datasets/generate_from_forecaster.py --model-horizon 6 --real-insample 18

python datasets/generate_from_forecaster.py --model-horizon 1 --real-insample 12
python datasets/generate_from_forecaster.py --model-horizon 2 --real-insample 12
python datasets/generate_from_forecaster.py --model-horizon 3 --real-insample 12
python datasets/generate_from_forecaster.py --model-horizon 4 --real-insample 12
python datasets/generate_from_forecaster.py --model-horizon 5 --real-insample 12

