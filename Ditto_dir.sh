python ../main.py  --purpose Cifar --device cpu --global_seed 0 --use_wandb False --yamlfile ./Cifar10_Conv2Cifar.yaml --strategy Ditto --num_clients 100 --participate_ratio 0.1 --partition noniid-label-distribution --beta 0.3 --num_rounds 200 --client_lr 0.01 --client_lr_scheduler diminishing --sgd_momentum 0.9 --sgd_weight_decay 1e-05 --num_epochs 5 --Ditto_lambda 0.75 &
python ../main.py  --purpose Cifar --device cpu --global_seed 0 --use_wandb False --yamlfile ./Cifar10_Conv2Cifar.yaml --strategy Ditto --num_clients 100 --participate_ratio 0.1 --partition noniid-label-distribution --beta 1.0 --num_rounds 200 --client_lr 0.01 --client_lr_scheduler diminishing --sgd_momentum 0.9 --sgd_weight_decay 1e-05 --num_epochs 5 --Ditto_lambda 0.75 &
python ../main.py  --purpose Cifar --device cpu --global_seed 0 --use_wandb False --yamlfile ./Cifar100_Conv2Cifar.yaml --strategy Ditto --num_clients 100 --participate_ratio 0.1 --partition noniid-label-distribution --beta 0.3 --num_rounds 200 --client_lr 0.01 --client_lr_scheduler diminishing --sgd_momentum 0.9 --sgd_weight_decay 1e-05 --num_epochs 5 --Ditto_lambda 0.75 &
python ../main.py  --purpose Cifar --device cpu --global_seed 0 --use_wandb False --yamlfile ./Cifar100_Conv2Cifar.yaml --strategy Ditto --num_clients 100 --participate_ratio 0.1 --partition noniid-label-distribution --beta 1.0 --num_rounds 200 --client_lr 0.01 --client_lr_scheduler diminishing --sgd_momentum 0.9 --sgd_weight_decay 1e-05 --num_epochs 5 --Ditto_lambda 0.75 &
