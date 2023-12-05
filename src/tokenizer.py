def cls_double_tokenize(tokenizer, item, **kwargs):
    ''' Tokenize given parent_text and text for classification task '''
    return tokenizer(f"{item['parent_text']}{tokenizer.eos_token}{tokenizer.bos_token}{item['text']}", **kwargs)

def cls_single_tokenize(tokenizer, item, **kwargs):
    ''' Tokenize given  text for classification task '''
    return tokenizer(f"{item['text']}", **kwargs)

def clm_tokenize(tokenizer, prompt, system_prompt, instruct_prompt, item, **kwargs):
    instruct_prompt = instruct_prompt.format(parent_text=item['parent_text'], text=item['text'])
    tokens = prompt.format(
        system_prompt=system_prompt, 
        instruct_prompt=instruct_prompt, 
    )
    return tokenizer.encode(tokens, **kwargs)

def clm_template_tokenize(tokenizer, turns, item, **kwargs):
    tokenizer.use_default_system_prompt = False

    for turn in turns:
        turn['content'] = turn['content'].format(**item)

    return tokenizer.apply_chat_template(turns, **kwargs)

def label_tokenize(tokenizer, labels, **kwargs):
    return tokenizer.encode(labels, add_special_tokens=False, **kwargs)[0]