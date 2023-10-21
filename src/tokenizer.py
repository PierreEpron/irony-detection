def cls_tokenize(tokenizer, parent_text, text, **kwargs):
    ''' Tokenize given parent_text and text for classification task '''
    return tokenizer(f"{parent_text}{tokenizer.eos_token}{tokenizer.bos_token}{text}", **kwargs)

def clm_tokenize(tokenizer, prompt, system_prompt, instruct_prompt, item, **kwargs):
    tokens = prompt.format(
        system_prompt=system_prompt, 
        instruct_prompt=instruct_prompt, 
        parent_text=item['parent_text'], 
        text=item['text']
    )
    return tokenizer.encode(tokens, **kwargs)
