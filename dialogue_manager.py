from abc import ABC, abstractmethod
import os
import ollama
import re
import logging
import sys
# Configure logging
logger = logging.getLogger('DialogueManager')
logger.setLevel(logging.DEBUG)

def logger_cfg():
    # Create handlers
    stdout_handler = logging.StreamHandler(sys.stdout)
    stderr_handler = logging.StreamHandler(sys.stderr)
    file_handler = logging.FileHandler('dialogue_manager.log')

    # Set levels for handlers
    stdout_handler.setLevel(logging.INFO)
    stderr_handler.setLevel(logging.DEBUG)
    file_handler.setLevel(logging.DEBUG)

    # Create formatters and add them to handlers
    stdout_formatter = logging.Formatter('%(asctime)s - %(message)s')
    stderr_formatter = logging.Formatter('\033[32m%(asctime)s - %(levelname)s - %(message)s\033[0m')
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    stdout_handler.setFormatter(stdout_formatter)
    stderr_handler.setFormatter(stderr_formatter)
    file_handler.setFormatter(file_formatter)

    # Add handlers to the logger
    logger.addHandler(stdout_handler)
    logger.addHandler(stderr_handler)
    logger.addHandler(file_handler)

def extract_action_and_argument(input_string):
    ## TODO add check in case there is more in the output string from the LLM
    # Define the regex pattern for extracting action and argument
    pattern = r'(\w+)\((\w+)\)'
    match = re.match(pattern, input_string)
    
    if match:
        action = match.group(1)  # Extract the action
        argument = match.group(2)  # Extract the argument
        return action, argument
    else:
        raise ValueError("Input string does not match the expected format 'action(argument)'.")

class DialogueST(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def update(self, parsed_input):
        pass

    @abstractmethod
    def is_order(self):
        pass


class BurgerST(DialogueST):
    def __init__(self):
        fields = [
            'patty_count',
            'cheese_count',
            'bacon_count',
            'tomato',
            'onions',
            'mayo',
            'ketchup',
            'cooking'
            ]
        self.order = {field : None for field in fields}

    def update(self, parsed_input: dict):
        for field in self.order:
            if field in parsed_input:
                if parsed_input[field] == 'null':
                    self.order[field] = None
                else:
                    self.order[field] = parsed_input[field]

    def is_order(self):
        for field in self.order:
            if self.order[field] is None:
                return False
        return True
    

class ConversationHistory():
    def __init__(self):
        self.msg_list = []
        self.actions = []
        self.roles = []

    def add(self, msg, role, action):
        self.roles.append(role)
        self.msg_list.append(msg)
        self.actions.append(action)

    def clear(self):
        self.msg_list = []
        self.actions = []
    
    def get_history(self):
        return self.msg_list
    
    def to_msg_history(self)->list[dict]:
        return [{'role': role, 'content': msg} for role, msg in zip(self.roles, self.msg_list)]

nlu_cfg = {
    'system_prompt_file' : 'nlu_prompt.txt',
    'model_name' : 'llama3',
}

dm_cfg = {
    'system_prompt_file' : 'dm_prompt.txt',
    'model_name' : 'llama3',
}

nlg_cfg = {
    'system_prompt_file' : 'nlg_prompt.txt',
    'model_name' : 'llama3',
}

import json
import re
from typing import Any, Union

def parse_json(json_str: str) -> Union[dict, list, None]:
    """
    Parses a JSON string into a Python dictionary or list, with robust handling for edge cases,
    including cleaning and extracting JSON from additional text.
    
    Args:
        json_str (str): A JSON-formatted string or text containing JSON.
    
    Returns:
        Union[dict, list, None]: Parsed JSON object (dict or list) or None if parsing fails.
    
    Raises:
        ValueError: If the input cannot be parsed into valid JSON after cleanup.
    """
    def clean_json_string(json_str: str) -> str:
        """
        Cleans the input string to make it more JSON-compliant and extracts the JSON portion.
        """
        # Strip leading/trailing whitespace and artifacts
        json_str = json_str.strip()
        
        # Extract JSON portion using regex
        json_pattern = r'({.*?}|\[.*?\])'  # Matches JSON objects ({}) or arrays ([])
        match = re.search(json_pattern, json_str, re.DOTALL)
        
        if match:
            return match.group(0)  # Return the matched JSON portion
        
        # If no valid JSON is found, return the string as-is (will fail during parsing)
        return json_str

    try:
        # Attempt to parse directly
        return json.loads(json_str)
    except json.JSONDecodeError:
        try:
            # Try cleaning the string and parsing again
            cleaned_json = clean_json_string(json_str)
            return json.loads(cleaned_json)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON string after cleanup: {e}") from e
    except TypeError as e:
        raise ValueError(f"Input must be a string: {e}") from e


class DialogueManager():
    def __init__(self, nlu_cfg, dm_cfg, nlg_cfg):
        self.nlu_cfg = nlu_cfg
        self.dm_cfg = dm_cfg
        self.nlg_cfg = nlg_cfg

        self.state_tracker = BurgerST() # TODO add vars
        self.history = ConversationHistory()
        self.welcome_msg = 'Hello! I am a burger ordering assistant. How can I help you today?\n'

    def start_conversation(self):
        self.history.add(self.welcome_msg, 'assistant', 'welcome')
        input_prompt = input(self.welcome_msg)
        self.history.add(input_prompt, 'user', 'input')
        meaning_representation = self.get_meaning_representation(input_prompt)
        self.state_tracker.update(meaning_representation)
        
        self.query_dialogue_manager(meaning_representation)

    def get_meaning_representation(self, input_prompt):
        raw_meaning_rep = self.query_model(self.nlu_cfg['model_name'], self.nlu_cfg['system_prompt_file'], input_prompt)
        try:
            meaning_representation = parse_json(raw_meaning_rep)
        except:
            logger.debug('\033[91m' + 'Error in parsing the meaning representation. Please try again.\n\n'
                         + raw_meaning_rep)
            return
        logger.debug(meaning_representation)
        
        return meaning_representation

    def ask_info(self, info):
        lexicalised_question = self.lexicalise('ask_info({})'.format(info))
        self.history.add(lexicalised_question, 'assistant', 'ask_info')
        
        input_prompt = input(lexicalised_question + '\n')
        self.history.add(input_prompt, 'user', 'input')

        meaning_representation = self.get_meaning_representation(input_prompt)
        self.state_tracker.update(meaning_representation)

        self.query_dialogue_manager(meaning_representation)

    def query_dialogue_manager(self, meaning_representation):
        """
        Query the dialogue manager model to get the next best action to perform.
        """
        raw_action = self.query_model(self.dm_cfg['model_name'], self.dm_cfg['system_prompt_file'], str(meaning_representation))
        # action = parse_json(raw_action) # TODO what to parse? Do we need to parse?
        # Define a format for asking info
        logging.debug(raw_action)
        try:
            action, argument = extract_action_and_argument(raw_action)
        except:
            logger.debug('\033[91m' + 'Error in parsing the action. Please try again.\n\n'
                         + raw_action)
            return
        
        if 'info' in action:
            self.ask_info(argument)
        elif 'confirm' in action:
            self.confirm_order()

    def confirm_order(self, order):
        lexicalised_ans = self.lexicalise('confirm_order\n' + order)
        self.history.add(lexicalised_ans, 'assistant', 'confirm_order')
        logging.debug(lexicalised_ans)

    def lexicalise(self, action):
        lexicalised_text = self.query_model(self.nlg_cfg['model_name'], self.nlg_cfg['system_prompt_file'], action)
        
        return lexicalised_text
    
    def query_model(self, model_name, system, input_text, max_seq_len=128):
        system_prompt = open(system, 'r').read()
        user_env = os.getenv('USER')
        if user_env == 'amir.gheser':
            # we are on the cluster
            pass
        elif user_env == 'amirgheser':
            # we are on the local machine
            messages = [{'role':'system', 'content': system_prompt}] + self.history.to_msg_history()

            response = ollama.chat(model=model_name, messages=
                messages
            )
            return response['message']['content']
        else:
            raise ValueError('Unknown user environment. Please set the USER environment variable.')


if __name__ == '__main__':

    dm = DialogueManager(nlu_cfg, dm_cfg, nlg_cfg)
    # response = dm.query_model('llama2', 'nlu_prompt.txt', 'I want a burger')
    # print(response['message']['content'])
    dm.start_conversation()
    # dm.ask_info('patty_count')
    # dm.ask_info('cheese_count')
    # dm.ask_info('bacon_count')
    # dm.ask_info('tomato')
    # dm.ask_info('onions')
    # dm.ask_info('mayo')
    # dm.ask_info('ketchup')
    # dm.ask_info('cooking')
    # dm.confirm_order
    print(dm.history.to_msg_history())