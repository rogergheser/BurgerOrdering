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

    @abstractmethod
    def to_dict(self):
        pass

    @abstractmethod
    def __str__(self):
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
        self.sentiment = None

    def update(self, parsed_input: dict):
        """
        Updates the dialogue state for the slots from the parsed input to the order state tracker.
        :param parsed_input: dict, parsed slots dictionary from the user.
        """
        if 'intent' in parsed_input:
            if parsed_input['intent'] != 'burger_ordering':
                logging.warning('Intent is not burger_ordering.')
                return
        if 'sentiment' in parsed_input:
            self.sentiment = parsed_input['sentiment']
        
        if 'slots' not in parsed_input:
            logging.warning('No slots found in parsed input.')
            return
        
        parsed_input = parsed_input['slots']
        for field in parsed_input:
            if parsed_input[field] == 'null':
                continue
            if field in self.order:
                self.order[field] = parsed_input[field]
            else:
                logging.warning(f'Field {field} not found in order fields.')

    def __str__(self):
        return ', '.join([f'{key}: {value}' for key, value in self.order.items()])

    def to_dict(self):
        return {"intent": "burger_ordering",
                "slots" : self.order,
                "sentiment" : self.sentiment}

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
    
    def action_history_str(self):
        return ', \n'.join(self.actions)

    def to_msg_history(self)->list[dict]:
        history = [{'role': role, 'content': msg} for role, msg in zip(self.roles, self.msg_list)]
        if len(history) > 5:
            return history[-5:]
        return history


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

class ColorFormatter(logging.Formatter):
    COLORS = {
        "white": "\033[97m",
        "red": "\033[91m",
        "green": "\033[92m",
        "blue": "\033[94m",
        "reset": "\033[0m"
    }

    def __init__(self, fmt, default_color="white"):
        super().__init__(fmt)
        self.default_color = self.COLORS.get(default_color.lower(), self.COLORS["white"])

    def format(self, record):
        # Use the color passed in `record.color` or fall back to the default color
        color = self.COLORS.get(getattr(record, 'color', '').lower(), self.default_color)
        msg = super().format(record)
        return f"{color}{msg}{self.COLORS['reset']}"


def logger_cfg(logger, debug_color="red", info_color="white"):
    logger.setLevel(logging.DEBUG)

    # File handler for debug logs
    debug_file_handler = logging.FileHandler('debug.log')
    debug_file_handler.setLevel(logging.DEBUG)
    debug_file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    debug_file_handler.setFormatter(debug_file_formatter)

    # File handler for info logs
    info_file_handler = logging.FileHandler('info.log')
    info_file_handler.setLevel(logging.INFO)
    info_file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    info_file_handler.setFormatter(info_file_formatter)

    # Console handler for debug logs (colorized)
    console_handler_debug = logging.StreamHandler()
    console_handler_debug.setLevel(logging.DEBUG)
    console_handler_debug.setFormatter(ColorFormatter('%(levelname)s - %(message)s', default_color=debug_color))

    # Console handler for info logs (colorized)
    console_handler_info = logging.StreamHandler()
    console_handler_info.setLevel(logging.INFO)
    console_handler_info.setFormatter(ColorFormatter('%(levelname)s - %(message)s', default_color=info_color))

    # Add handlers to logger
    logger.addHandler(debug_file_handler)
    logger.addHandler(info_file_handler)
    logger.addHandler(console_handler_debug)
    logger.addHandler(console_handler_info)

def extract_action_and_argument(input_string):
    ## TODO add check in case there is more in the output string from the LLM
    # Remove any ' characters from the input string
    input_string = input_string.replace("'", "")
    input_string = input_string.replace("\"", "")
    # Define the regex pattern for extracting action and argument
    pattern = r'(\w+)\((\w+)\)'
    match = re.match(pattern, input_string)
    
    if match:
        action = match.group(1)  # Extract the action
        argument = match.group(2)  # Extract the argument
        return action, argument
    else:
        raise ValueError("Input string does not match the expected format 'action(argument)'.")
