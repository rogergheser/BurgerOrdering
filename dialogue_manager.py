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
        self.RUNNING = True
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
        
        while self.RUNNING:
            # NLU
            meaning_representation = self.get_meaning_representation(input_prompt)
            self.state_tracker.update(meaning_representation)
            logger.info(self.state_tracker, extra={"color": "green"})
            # DM
            action, argument = self.query_dialogue_manager(self.state_tracker)
            lexicalise_query = self.nba_handler(action, argument)   
            logger.info(f'Action: {action}, Argument: {argument}', extra={"color": "green"})
            
            if self.RUNNING:
                # NLG
                lexicalised_query = self.lexicalise(lexicalise_query)
                self.history.add(lexicalised_query, 'assistant', action)
                
                input_prompt = input(lexicalised_query + '\n')
                self.history.add(input_prompt, 'user', 'input')

    def nba_handler(self, action, argument)->str:
        match action:
            case 'ask_info':
                return f'ask_info({argument})'
            case 'request_info':
                return f'ask_info({argument})'
            case 'confirm_order':
                return self.confirm_order()
            case 'confirmation':
                return self.confirm_order()
            case _:
                raise ValueError(f'Unknown action: {action}')

    def confirm_order(self):
        print('\033[94m' + "Confirming order")
        order = str(self.state_tracker.to_dict())
        lexicalised_ans = self.lexicalise('confirm_order\n' + order)
        self.history.add(lexicalised_ans, 'assistant', 'confirm_order')
        
        input_prompt = input('\033[96m' + lexicalised_ans + '\nYes/No\n')

        if input_prompt.lower() == 'yes' or input_prompt.lower() == 'y':
            self.history.add('yes', 'user', 'confirm_order')
            self.history.add('confirmed', 'assistant', 'confirm_order')
            self.RUNNING = False
            print('\033[96m' + 'Thank you for ordering with us!')
        elif input_prompt.lower() == 'no' or input_prompt.lower() == 'n':
            self.history.add('no', 'user', 'confirm_order')
            self.history.add('not confirmed', 'assistant', 'confirm_order')
        else:
            logger.warning("Only 'yes' or 'no' inputs are accepted.")
            self.history.add(input_prompt, 'user', 'confirm_order')

        return input_prompt.lower()


    def get_meaning_representation(self, input_prompt):
        raw_meaning_rep = self.query_model(self.nlu_cfg['model_name'], self.nlu_cfg['system_prompt_file'], input_prompt)
        try:
            meaning_representation = parse_json(raw_meaning_rep)
        except:
            logger.debug('\033[91m' + 'Error in parsing the meaning representation. Please try again.\n\n'
                         + raw_meaning_rep)
            return 
        logger.info(meaning_representation)
        
        return meaning_representation

    def query_dialogue_manager(self, state_tracker: DialogueST):
        """
        Query the dialogue manager model to get the next best action to perform.
        """
        meaning_representation = state_tracker.to_dict()
        raw_action = self.query_model(self.dm_cfg['model_name'], self.dm_cfg['system_prompt_file'], str(meaning_representation))
        # action = parse_json(raw_action) # TODO what to parse? Do we need to parse?
        # Define a format for asking info
        logger.debug(raw_action)
        try:
            action, argument = extract_action_and_argument(raw_action)
        except:
            logger.debug('\033[91m' + 'Error in parsing the action. Please try again.\n\n'
                         + raw_action)
            return
        
        return action, argument

    def lexicalise(self, action):
        logger.debug('lexicalise: ' + action)
        lexicalised_text = self.query_model(self.nlg_cfg['model_name'], self.nlg_cfg['system_prompt_file'], input_text=action)

        return lexicalised_text
    
    def query_model(self, model_name, system, input_text=False, max_seq_len=128):
        system_prompt = open(system, 'r').read()
        user_env = os.getenv('USER')
        if user_env == 'amir.gheser':
            # we are on the cluster
            pass
        elif user_env == 'amirgheser':
            # we are on the local machine
            messages = [{
                            'role':'system',
                            'content': system_prompt
                            }] + self.history.to_msg_history()
                        # + [{
                        #     'role':'system',
                        #     'content': system_prompt
                        #     }]
            if input_text:
                messages.append({
                    'role': 'user',
                    'content': input_text
                })
            logger.debug(messages, extra={"color": "blue"})
            response = ollama.chat(model=model_name, messages=
                messages
            )
            return response['message']['content']
        else:
            raise ValueError('Unknown user environment. Please set the USER environment variable.')

    # def ask_info(self, info):
    #     lexicalised_question = self.lexicalise('ask_info({})'.format(info))
    #     self.history.add(lexicalised_question, 'assistant', 'ask_info')
        
    #     input_prompt = input(lexicalised_question + '\n')
    #     self.history.add(input_prompt, 'user', 'input')

    #     meaning_representation = self.get_meaning_representation(input_prompt)
    #     self.state_tracker.update(meaning_representation)
    #     logger.info(self.state_tracker, extra={"color": "green"})

    #     self.query_dialogue_manager(self.state_tracker)
    # def confirm_order(self, order):
    #     # print in blue
    #     print('\033[94m' + "Confirming order")
    #     lexicalised_ans = self.lexicalise('confirm_order\n' + order)
    #     self.history.add(lexicalised_ans, 'assistant', 'confirm_order')
    #     logging.debug(lexicalised_ans)

nlu_cfg = {
    'system_prompt_file' : 'prompts/nlu_prompt.txt',
    'model_name' : 'llama3',
}

dm_cfg = {
    'system_prompt_file' : 'prompts/dm_prompt.txt',
    'model_name' : 'llama3',
}

nlg_cfg = {
    'system_prompt_file' : 'prompts/nlg_prompt.txt',
    'model_name' : 'llama3',
}


if __name__ == '__main__':
    logger_cfg(logger)
    dm = DialogueManager(nlu_cfg, dm_cfg, nlg_cfg)
    dm.start_conversation()

    print(dm.history.to_msg_history())