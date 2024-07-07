for seed in 38 39 40 41 42
do
    CUDA_VISIBLE_DEVICES=1 python3 icl.py --data snli --seed $seed  --n_shot 8 --max_train_size 10000 --max_dev_size 500 --selection "random_stratify_sampling" --llm llama2_13b
done

CUDA_VISIBLE_DEVICES=1 python3 icl.py --data snli --seed 42 --n_shot 8 --max_train_size 10000 --max_dev_size 500 --selection similar --llm llama2_13b
CUDA_VISIBLE_DEVICES=1 python3 icl.py --data snli --seed 42 --n_shot 8 --max_train_size 10000 --max_dev_size 500 --selection pair_similar --llm llama2_13b
CUDA_VISIBLE_DEVICES=1 python3 icl.py --data snli --seed 42 --n_shot 8 --max_train_size 10000 --max_dev_size 500 --selection pair_one_similar --llm llama2_13b
CUDA_VISIBLE_DEVICES=1 python3 icl.py --data snli --seed 42 --n_shot 8 --max_train_size 10000 --max_dev_size 500 --selection pair_one_similar2 --llm llama2_13b