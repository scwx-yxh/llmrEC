"""
A dedicated helper to manage templates and prompt building.
"""

import json
import os.path as osp
from typing import Union, List


class Prompter(object):
    def __init__(self, template_name: str = ""):
        if not template_name:
            # Enforce the default here, so the constructor can be called with '' and will not break.
            template_name = "alpaca"
        file_name = osp.join("templates", f"{template_name}.json")
        if not osp.exists(file_name):
            raise ValueError(f"Can't read {file_name}")
        with open(file_name) as fp:
            self.template = json.load(fp)

    def generate_prompt(self, task_type: str,) -> List[str]:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        # if task_type == 'general':
        #     instruction = "Given the user ID and purchase history, predict the most suitable item for the user."
        # elif task_type == 'sequential':
        #     instruction = "Given the user’s purchase history, predict next possible item to be purchased."
        # else:
        #     instruction = ""
        # if self.modal == "true":
        #     instruction = "Given the user ID and interaction history, each item is accompanied by "
        #     if self.modal_num == 2:
        #         instruction += "two"
        #     elif self.modal_num == 3:
        #         instruction += "three"
        #     instruction += " modal information, predict the most suitable item for the user."
        # else:
        instruction="Given the user ID and interaction history, predict the user's preferences, ensuring that recommendations are diverse and aligned with the user's past interactions."
            # instruction = "Given the user ID and interaction history, predict the most suitable item for the user."
        
        ins = self.template["prompt_input"].format(
            instruction=instruction
        )
        res = self.template["response_split"]
        return [ins, res]

