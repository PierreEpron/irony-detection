from copy import deepcopy
import json
from pathlib import Path
import random

TURNS_TEMPLATE = [
    {"role": "system", "content": "{system}"},
    {"role": "user", "content": '{user}'},
    {"role": "assistant", "content": '{assistant}'}
]

def randomize_phrases(phrases):
    """
        ...
        Parameters
        ==========
        
        Returns
        =======
        
        Examples
        ========
        >>> random.seed(0)
        >>> randomize_phrases({'A':[{'text':0}, {'text':1}, {'text':2}], 'B':[{'text':'C'}]})
        ({'A': 1, 'B': 'C'}, {'A': 1, 'B': 0})
        >>> randomize_phrases({'A':[{'text':0}, {'text':1}, {'text':2}], 'B':[{'text':'C'}]})
        ({'A': 0, 'B': 'C'}, {'A': 0, 'B': 0})
        >>> randomize_phrases({'A':[{'text':0}, {'text':1}, {'text':2}], 'B':[{'text':'C'}]})
        ({'A': 2, 'B': 'C'}, {'A': 2, 'B': 0})
    """
    # We may want to track indice for evaluation.
    indices = {k:random.randrange(len(v)) for k, v in phrases.items()}
    return {k:v[indices[k]]['text'] for k, v in phrases.items()}, indices


def format_closure(text, item):
    """
        ...
        Parameters
        ==========
        
        Returns
        =======
        
        Examples
        ========
        >>> format_closure('0{A}1', {'A':'0{AA}1', 'AA':'2'})
        '00211'
    """
    _text = None
    while _text != text:
        _text = text
        text = text.format(**item)
    return text

def generate_turns(item, phrases, tempate=TURNS_TEMPLATE):
    """
        ...
        Parameters
        ==========
        
        Returns
        =======
        
        Examples
        ========
        >>> random.seed(42)
        >>> phrases = {'system':[{'text': 'Some names:'}, {'text':'Names:'}], 'user':[{'text':'{firstname} {name}'}, {'text':'Anonyme'}],'firstname':[{'text':'Noam'}, {'text':'Richard'}, {'text':'Alan'}],'name':[{'text':'Chomsy'}, {'text':'Montague'}, {'text':'Turing'}],"assistant":[{'text': 'End'}],}
        >>> generate_turns({}, phrases=phrases)
        ([{'role': 'system', 'content': 'Some names:'}, {'role': 'user', 'content': 'Alan Montague'}, {'role': 'assistant', 'content': 'End'}], {'system': 0, 'user': 0, 'firstname': 2, 'name': 1, 'assistant': 0})
    """
    turns = deepcopy(tempate)
    random_phs, seed_phs = randomize_phrases(phrases)
    new_item = item | random_phs
    for turn in turns:
        turn['content'] = format_closure(turn['content'], new_item)
    return turns, seed_phs 


def load_phrases(path):
    path = Path(path) if isinstance(path, str) else path
    phrases = json.loads(path.read_text())
    return phrases, phrases['labels'][0]['values']


if __name__ == "__main__":
    import doctest
    doctest.testmod()