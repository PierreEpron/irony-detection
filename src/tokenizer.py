def cls_tokenize(tokenizer, parent_text, text, **kwargs):
    ''' Tokenize given parent_text and text for classification task '''
    return tokenizer(f"{parent_text}{tokenizer.eos_token}{tokenizer.bos_token}{text}", **kwargs)
