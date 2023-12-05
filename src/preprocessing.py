from pathlib import Path
import pandas as pd
import random
import re
import demoji

from src.tokenizer import cls_double_tokenize

from sklearn.model_selection import StratifiedShuffleSplit

from src.utils import read_jsonl, find_closest

REG_EXPRS = {'w':('(https?:\/\/)?([\w\d\-_]+)\.([\w\d\-_]+)\/?\??([^ \.#\n\r]*)?#?([^ \n\r]*)', ''), 'u':('@[\w]+', ''), 'n':('\n', ' ')}

def make_line(values:pd.DataFrame, equality, cleaning, emojis, new_labels):
    """
        Returns a list of values based on a dataframe containing annotation for one Post/Reply pair.

        Parameters
        ----------
        values : pandas dataframe.
        equality : whether pairs where there is no majority decision for the label should be kept.
        cleaning : list of things that should be removed from the texts. Values must be n (newlines), w (weblinks) or u (user mentions).
        emojis : to handle emojis. Values must be either 'rem' (remove), 'keep' or a str to be used as separator between their description.
        new_labels : list of new labels to use. Must be listed from not ironic to ironic.

        Returns
        ---------
        line : a list with the source, subreddit, id_original, text, parent_id_original, parent_text, Language_instance and the label (0 : not irony/ 1 : irony).
    """
    
    line = values.iloc[0,3:10].values.tolist() #keep the values that are shared
    labels = values.label.value_counts().to_dict()

    if len(labels)>1 and labels['iro'] == labels['not']: #if this is an equality 
        if not equality: #if equalities are not kept
            return None
        else: #if equalities are kept
            label = random.randint(0, 1)
    else: #if this is not an equality
        # label = 1 if pd.Series(labels).idxmax()=='iro' else 0 #set the label to 1 if text is irony and 0 otherwise
        label = new_labels[find_closest(labels.get('iro', 0)/(labels.get('iro', 0)+labels.get('not', 0)), list(new_labels.keys()))]

    if emojis=='rem':
        line[3] = demoji.replace(line[3], '')
        line[5] = demoji.replace(line[5], '')
    elif emojis!='keep':
        line[3] = demoji.replace_with_desc(line[3], emojis)
        line[5] = demoji.replace_with_desc(line[5], emojis)

    for pattern in cleaning:
        line[3] = re.sub(REG_EXPRS[pattern][0], REG_EXPRS[pattern][1], line[3])
        line[5] = re.sub(REG_EXPRS[pattern][0], REG_EXPRS[pattern][1], line[5])

    line.append(label)
    
    return line

def make_dataset(dataset, n_annotators:int = None, equality = False, cleaning:list = ['n'], emojis:str = ' ', new_labels:list = [0,1]) -> pd.DataFrame:
    """
        Returns a dataset of Post/Reply pairs with a 1 (irony) or 0 (not irony) label.

        Parameters
        ----------
        dataset : pandas dataframe.
        n_annotators : minimum number of annotators for each pair.
        equality : whether pairs where there is no majority decision for the label should be kept.
        cleaning : list of things that should be removed from the texts. Values must be n (newlines), w (weblinks) or u (user mentions).
        emojis : to handle emojis. Values must be either 'rem' (remove), 'keep' or a str to be used as separator between their description.
        new_labels : list of new labels to use. Must be listed from not ironic to ironic.

        Returns
        ---------
        prepared_dataset : a dataset with 0/1 labels for each Post/Reply pair. 
    """

    tab = []
    new_labels = {x/(len(new_labels)-1):y for x,y in enumerate(new_labels, start=0)}
    
    for id, values in dataset.groupby('id_original'):

        line = None #reset line

        if not n_annotators or (n_annotators and len(values.label)>=n_annotators): #if no amount of min annotators is provided OR an amount of min annotators is provided and respected
            line = make_line(values, equality, cleaning, emojis, new_labels) #attempts to create a line

        if line: #if a line was created
            tab.append(line)
            
    return pd.DataFrame(tab, columns=dataset.columns[3:10].tolist()+['label'])

def filter_dataset(df, tokenizer, max_token=514):

    df['n_tokens'] = df.apply(lambda x: len(cls_double_tokenize(tokenizer, x.parent_text, x.text).input_ids), axis = 1).squeeze()

    return df[df.n_tokens<max_token]

def split_dataset(df, n_splits=5, test_size=0.4, train_size=0.6, random_state=0):

    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, train_size=train_size, random_state=random_state)

    list_comp = []
    for train, val_test in sss.split(df,df.label):
        df_val_test = df.iloc[val_test]

        val_iro = df_val_test[df_val_test.label==1][::2].id_original.to_list()
        test_iro = df_val_test[df_val_test.label==1][1::2].id_original.to_list()

        val_not_iro = df_val_test[df_val_test.label==0][::2].id_original.to_list()
        test_not_iro = df_val_test[df_val_test.label==0][1::2].id_original.to_list()
        
        test = test_iro+test_not_iro
        val = val_iro+val_not_iro

        list_comp.append({'train':df.iloc[train].id_original.tolist(),'val':val,'test':test})

    return list_comp

def recover_split(split, df):
    return df[df['id_original'].isin(split)].to_dict(orient='records')

def iter_splits(splits_path, df):
    splits = read_jsonl(splits_path)
    for split in splits:
        yield recover_split(split['train'], df), recover_split(split['val'], df), recover_split(split['test'], df)

def load_tweeteval_set(name, path):
    X = (path / f'{name}_text.txt').read_text().split('\n')
    Y = (path / f'{name}_labels.txt').read_text().split('\n')
    return [{'text':x, 'label':y} for x, y in zip(X,Y)]

def load_tweeteval(path='data/tweet-eval/'):
    path = Path(str(path)) 
    return (
        load_tweeteval_set('train', path),
        load_tweeteval_set('val', path),
        load_tweeteval_set('test', path)
    )
