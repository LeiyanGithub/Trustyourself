python main.py --dataset hotpotqa --model gpt-3.5-turbo --task rewrite --promptfile myprompt
# python main.py --dataset hotpotqa --model gpt-3.5-turbo --task rewrite --promptfile myprompt

# python main.py --dataset hotpotqa --model llama-2-7b-chat-hf --promptfile myprompt
# python main.py --dataset hotpotqa --model llama-2-7b-chat-hf --task rewrite --promptfile myprompt


# python main.py --dataset hotpotqa --model llama-2-13b-chat-hf --promptfile myprompt
# python main.py --dataset hotpotqa --model llama-2-13b-chat-hf --task rewrite --promptfile myprompt

# python preprocess.py --dataset hotpotqa --model llama-2-7b-chat-hf --task rewrite --promptfile myprompt



# CUDA_VISIBLE_DEVICES=1 python batch_generate.py \
#     --model_name  llama-2-7b-chat-hf \
#     --model_path /home/liulian/yan/pytorch_model/llama-2-7b-chat-hf \
#     --task rewrite \

# CUDA_VISIBLE_DEVICES=0 python batch_generate.py --model_name  llama-2-13b-chat-hf --model_path /home/liulian/yan/pytorch_model/llama-2-13b-chat-hf --task rewrite

# CUDA_VISIBLE_DEVICES=1s python batch_generate.py \
#     --model_name  llama-2-7b-chat-hf \
#     --model_path /home/liulian/yan/pytorch_model/llama-2-7b-chat-hf \
#     --task rewrite \

# CUDA_VISIBLE_DEVICES=1s python batch_generate.py \
#     --model_name  llama-2-7b-chat-hf \
#     --model_path /home/liulian/yan/pytorch_model/llama-2-7b-chat-hf \
#     --task rewrite \
