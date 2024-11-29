import json
import re
import logging
import sys
from typing import Any, Union
from abc import ABC, abstractmethod
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


def logger_cfg(logger):
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
