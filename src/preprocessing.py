import pandas as pd
import random

def make_line(values:pd.DataFrame, equality):
    """
        Returns a list of values based on a dataframe containing annotation for one Post/Reply pair.

        Parameters
        ----------
        values : pandas dataframe.
        equality : whether pairs where there is no majority decision for the label should be kept.

        Returns
        ---------
        line : a list with the source, subreddit, id_original, text, parent_id_original, parent_text, Language_instance and the label (0 : not irony/ 1 : irony).
    """
    
    line = values.iloc[0,3:10].values.tolist() #keep the values that are shared
    labels = values.label.value_counts()

    if len(labels)>1 and labels.iloc[0] == labels.iloc[1]: #if this is an equality 
        if not equality: #if equalities are not kept
            return None
        elif equality=='iro': #if equalities default to irony
            label = 1
        elif equality=='not': #if equalities default to not irony
            label = 0
        elif equality: #if equalities are kept
            label = random.randint(0, 1)
    else: #if this is not an equality
        label = 1 if labels.idxmax()=='iro' else 0 #set the label to 1 if text is irony and 0 otherwise
    
    
    line.append(label)
    
    return line

def make_dataset(dataset, n_annotators=None, equality=False) -> pd.DataFrame:
    """
        Returns a dataset of Post/Reply pairs with a 1 (irony) or 0 (not irony) label.

        Parameters
        ----------
        dataset : pandas dataframe.
        n_annotators : minimum number of annotators for each pair.
        equality : whether pairs where there is no majority decision for the label should be kept.

        Returns
        ---------
        prepared_dataset : a dataset with 0/1 labels for each Post/Reply pair. 
    """

    tab_majority_decision = []
    
    for id, values in dataset.groupby('id_original'):

        line = None #reset line

        if not n_annotators or (n_annotators and len(values.label)>=n_annotators): #if no amount of min annotators is provided OR an amount of min annotators is provided and respected
            line = make_line(values, equality) #attempts to create a line

        if line: #if a line was created
            tab_majority_decision.append(line)
            
    return pd.DataFrame(tab_majority_decision, columns=dataset.columns[3:10].tolist()+['label'])