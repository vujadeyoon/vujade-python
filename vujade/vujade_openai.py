"""
Dveloper: vujadeyoon
Email: vujadeyoon@gmail.com
Github: https://github.com/vujadeyoon/vujade

Title: vujade_openai.py
Description: A module for OpenAI
"""


import argparse
import openai
from typing import Optional
from vujade.vujade_debug import printd


class ChatGPT(object):
    def __init__(self, _openai_api_key: str, _model_engine: str = 'text-davinci-002') -> None:
        super(ChatGPT, self).__init__()
        self.openai_api_key = _openai_api_key
        self.model_engine = _model_engine
        self._set_openai_api_key()

    def generate_response(self, _prompt: str) -> str:
        response = openai.Completion.create(
            engine=self.model_engine,
            prompt=_prompt,
            max_tokens=1024,
            n=1,
            stop=None,
            temperature=0.7
        )

        ans = response["choices"][0]["text"].strip()

        return ans

    def _set_openai_api_key(self) -> None:
        openai.api_key = self.openai_api_key


class MainChatGPT(object):
    @classmethod
    def run(cls) -> None:
        args = cls._get_args()

        chatgpt = ChatGPT(_openai_api_key=args.openai_aip_key)

        ans = chatgpt.generate_response(_prompt="Let's have a chat.")
        printd('ChatGPT: {}'.format(ans), _is_pause=False)

        while True:
            prompt = input("You: ")
            ans = chatgpt.generate_response(_prompt=prompt)
            printd('ChatGPT: {}'.format(ans), _is_pause=False)

    @staticmethod
    def _get_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser(description='Main function for calling OpenAI ChatGPT.')
        parser.add_argument('--openai_api_key', type=str, required=True)
        args = parser.parse_args()

        return args
