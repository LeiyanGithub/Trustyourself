import openai
import ast
import asyncio
from typing import Any, List, Dict

import openai


class OpenAIChat():
    def __init__(
            self,
            model_name='gpt-3.5-turbo',
            max_tokens=100,
            temperature=0.3,
            top_p=1,
            request_timeout=600,
    ):
        if 'gpt' not in model_name:
            openai.api_key = "EMPTY"
            openai.api_base = "http://127.0.0.1:8000/v1"
        else:
            openai.api_base = "https://one.aiskt.com/v1"
            openai.api_key = 'sk-Bd2ysg0m6cQ4qVxm393aDb4720F14c10Af9cEdBf35A120Ba'


        self.config = {
            'model_name': model_name,
            'max_tokens': max_tokens,
            'temperature': temperature,
            'top_p': top_p,
            'request_timeout': request_timeout,  
        }

    def generate(self, messages):
        response = openai.ChatCompletion.create(
            model=self.config['model_name'],
            messages=messages,
            max_tokens=self.config['max_tokens'],
            temperature=self.config['temperature'],
            top_p=self.config['top_p'],
            )
        preds = response['choices'][0]['message']['content']
        print("preds: ", preds)
        return preds
    



    async def dispatch_openai_requests(
        self,
        messages_list,
    ) -> list[str]:
        """Dispatches requests to OpenAI API asynchronously.
        
        Args:
            messages_list: List of messages to be sent to OpenAI ChatCompletion API.
        Returns:
            List of responses from OpenAI API.
        """
        async def _request_with_retry(messages, retry=3):
            for _ in range(retry):
                # try:
                    print("messages:", messages)
                    response = openai.ChatCompletion.create(
                        model=self.config['model_name'],
                        messages=messages,
                        max_tokens=self.config['max_tokens'],
                        temperature=self.config['temperature'],
                        top_p=self.config['top_p'],
                    )

                    return response['choices'][0]['message']['content']

        async_responses = [
            _request_with_retry(messages)
            for messages in messages_list
        ]

        return await asyncio.gather(*async_responses)
    
    async def async_run(self, messages_list):

        retry = 3
        responses = [None for _ in range(len(messages_list))]
        messages_list_cur_index = [i for i in range(len(messages_list))]

        while retry > 0 and len(messages_list_cur_index) > 0:
            try:
                print(f'{retry} retry left...')
                messages_list_cur = [messages_list[i] for i in messages_list_cur_index]
                
                predictions = await self.dispatch_openai_requests(
                    messages_list=messages_list_cur,
                )

                finised_index = []
                for i, pred in enumerate(predictions):
                    if pred is not None:
                        responses[messages_list_cur_index[i]] = pred
                        finised_index.append(messages_list_cur_index[i])
                
                messages_list_cur_index = [i for i in messages_list_cur_index if i not in finised_index]
                return responses
            except Exception as e:
                print(e)
                retry -= 1

if __name__ == "__main__":

    chat = OpenAIChat(model_name='llama-2-7b-chat-hf')
    predictions = chat.generate([{"role": "user", "content": "假设你是李白，自由洒脱的诗人，请你按照李白的性格回答下列问题，要求：不要有废话，直接给出答案。当你要外出一整天时，你会：\\nA.计划要做的事情以及做的时间\\nB.不做计划，说走就走"}])
    print(predictions)