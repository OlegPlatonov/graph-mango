python train.py --name test_run --dataset tolokers --model ResNet --device cuda:0 --amp
python train.py --name test_run --dataset tolokers --model GraphSAGE --device cuda:0 --amp
python train.py --name test_run --dataset tolokers-tab --model ResNet --numerical_features_transform quantile-transform-normal --plr --device cuda:0 --amp
python train.py --name test_run --dataset tolokers-tab --model GraphSAGE --numerical_features_transform quantile-transform-normal --plr --device cuda:0 --amp
python train.py --name test_run --dataset hm-categories --model ResNet --numerical_features_transform quantile-transform-normal --plr --device cuda:0 --amp
python train.py --name test_run --dataset hm-categories --model GraphSAGE --numerical_features_transform quantile-transform-normal --plr --device cuda:0
python train.py --name test_run --dataset hm-categories --model GT-sep --numerical_features_transform quantile-transform-normal --plr --device cuda:0 --amp
python train.py --name test_run --dataset hm-prices --model ResNet --dropout 0 --lr 3e-5 --regression_target_transform standard-scaler --device cuda:0 --amp
python train.py --name test_run --dataset hm-prices --model GraphSAGE --dropout 0 --lr 3e-5 --regression_target_transform standard-scaler --device cuda:0 --amp
python train.py --name test_run --dataset hm-prices --model GT --dropout 0 --lr 3e-5 --regression_target_transform standard-scaler --device cuda:0 --amp
python train.py --name test_run --dataset city-roads-M --model ResNet --dropout 0 --lr 3e-3 --numerical_features_transform quantile-transform-normal --device cuda:0 --amp
python train.py --name test_run --dataset city-roads-M --model GraphSAGE --dropout 0 --lr 3e-3 --numerical_features_transform quantile-transform-normal --device cuda:0 --amp
python train.py --name test_run --dataset amazon-ratings --model ResNet --num_steps 2500 --device cuda:0 --amp
python train.py --name test_run --dataset amazon-ratings --model GraphSAGE --num_steps 2500 --device cuda:0 --amp
python train.py --name test_run --dataset ogbn-products --model ResNet --num_layers 2 --device cuda:0 --amp
python train.py --name test_run --dataset ogbn-products --model GraphSAGE --num_layers 2 --device cuda:0 --amp
python train.py --name test_run --dataset web-fraud --model ResNet --num_layers 2 --hidden_dim 256 --numerical_features_transform quantile-transform-normal --device cuda:0
python train.py --name test_run --dataset web-fraud --model GraphSAGE --num_layers 2 --hidden_dim 256 --numerical_features_transform quantile-transform-normal --device cuda:0
python train.py --name test_run --dataset web-traffic --model ResNet --num_layers 2 --hidden_dim 256 --numerical_features_transform quantile-transform-normal --device cuda:0
python train.py --name test_run --dataset web-traffic --model GraphSAGE --num_layers 2 --hidden_dim 256 --numerical_features_transform quantile-transform-normal --device cuda:0
