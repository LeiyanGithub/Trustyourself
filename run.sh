# python main.py --dataset hotpotqa --model gpt-3.5-turbo --task rewrite --promptfile myprompt
# python main.py --dataset hotpotqa --model gpt-3.5-turbo --task rewrite --promptfile myprompt

# python main.py --dataset hotpotqa --model llama-2-7b-chat-hf --promptfile myprompt
# python main.py --dataset hotpotqa --model llama-2-7b-chat-hf --task rewrite --promptfile myprompt


# python main.py --dataset hotpotqa --model llama-2-13b-chat-hf --promptfile myprompt
# python main.py --dataset hotpotqa --model llama-2-13b-chat-hf --task rewrite --promptfile myprompt

# python preprocess.py --dataset hotpotqa --model llama-2-7b-chat-hf --task rewrite --promptfile myprompt



# CUDA_VISIBLE_DEVICES=0 python batch_generate.py \
#     --model_name  llama-2-7b-chat-hf \
#     --model_path /home/liulian/yan/pytorch_model/llama-2-7b-chat-hf \
#     --task rewrite \

# CUDA_VISIBLE_DEVICES=1 python batch_generate.py --model_name  llama-2-13b-chat-hf --model_path /home/liulian/yan/pytorch_model/llama-2-13b-chat-hf --task rewrite

# CUDA_VISIBLE_DEVICES=1 python batch_generate.py --model_name  Qwen1.5-7B-Chat --model_path /home/liulian/yan/pytorch_model/Qwen1.5-7B-Chat --task rewrite


# CUDA_VISIBLE_DEVICES=1s python batch_generate.py \
#     --model_name  llama-2-7b-chat-hf \
#     --model_path /home/liulian/yan/pytorch_model/llama-2-7b-chat-hf \
#     --task rewrite \

# CUDA_VISIBLE_DEVICES=1s python batch_generate.py \
#     --model_name  llama-2-7b-chat-hf \
#     --model_path /home/liulian/yan/pytorch_model/llama-2-7b-chat-hf \
#     --task rewrite \


# python generate.py --dataset hotpotqa --target gpt-3.5-turbo --pos_num 2 --neg_num 0 --task qa --promptfile myprompt --nums 500 
# python generate.py --dataset hotpotqa --target gpt-3.5-turbo --pos_num 2 --neg_num 2 --task qa --promptfile myprompt --nums 500 
# python generate.py --dataset hotpotqa --target gpt-3.5-turbo --pos_num 0 --neg_num 2 --task qa --promptfile myprompt --nums 500 

# python evaluate_result.py --data_path ./results/hotpotqa/500/gpt-3.5-turbo_pos_2_neg_0.json
# python evaluate_result.py --data_path ./results/hotpotqa/500/gpt-3.5-turbo_pos_2_neg_2.json

CUDA_VISIBLE_DEVICES=1 python generate.py --dataset hotpotqa --target llama-2-7b-chat-hf --pos_num 2 --neg_num 0 --task qa --promptfile myprompt --nums 500 
# CUDA_VISIBLE_DEVICES=1 python generate.py --dataset hotpotqa --target llama-2-7b-chat-hf --pos_num 2 --neg_num 2 --task qa --promptfile myprompt --nums 500 
# python generate.py --dataset hotpotqa --target gpt-3.5-turbo --pos_num 0 --neg_num 2 --task qa --promptfile myprompt --nums 500 