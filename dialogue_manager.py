from abc import ABC, abstractmethod
import os
import ollama
import re
import logging
import sys
from utils import *

# Configure logging
logger = logging.getLogger('DialogueManager')
logger.setLevel(logging.DEBUG)

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


if __name__ == '__main__':
    dm = DialogueManager(nlu_cfg, dm_cfg, nlg_cfg)
    dm.start_conversation()
    logger_cfg(logger)

    print(dm.history.to_msg_history())