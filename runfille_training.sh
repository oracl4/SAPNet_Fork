# ITRI Dataset

## Training Baseline Network
python train.py \
--config-file configs/itri/baseline_training.yaml

## Testing Baseline Network
python train.py \
--config-file configs/itri/baseline_testing.yaml \
--resume debug/itri/baseline_training/model_epoch_14.pth \
--test-only

## Training Adversarial Network
python adversarial_train.py \
--config-file configs/itri/adversarial_training.yaml

# Resume
python adversarial_train.py \
--config-file configs/itri/adversarial_training.yaml \
--resume debug/itri/adversarial_training_v1/model_epoch_17.pth

## Testing Adversarial Network
python adversarial_train.py \
--config-file configs/itri/adversarial_testing.yaml \
--resume debug/itri/adversarial_training/best.pth \
--test-only

## Fine Tuning Adversarial Network
python train.py \
--config-file configs/itri/adversarial_finetune.yaml \
--resume debug/itri/adversarial_training/best.pth

# Fine Tune Testing
python train.py \
--config-file configs/itri/finetune_testing.yaml \
--resume debug/itri/adversarial_finetune/model_epoch_27.pth \
--test-only

## Training Adversarial Network Lambda : 0.35
python adversarial_train.py \
--config-file configs/itri/adversarial_training_lambda_35.yaml

# Training Oracle
python train.py \
--config-file configs/itri/oracle_training.yaml

# Testing Oracle
python train.py \
--config-file configs/itri/oracle_testing.yaml \
--resume debug/itri/oracle_training/model_epoch_21.pth \
--test-only