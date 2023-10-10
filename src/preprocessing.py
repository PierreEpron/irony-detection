import pandas as pd

def make_line(values, equality):
    
    line = values.iloc[0,3:10].values.tolist()
    labels = values.label.value_counts()

    if not equality and len(labels)>1 and labels[0] == labels[1]:
        return None
    
    label = 1 if labels.idxmax()=='iro' else 0
    line.append(label)
    
    return line

def make_dataset(dataset, n_annotators=None, equality=False) -> pd.DataFrame:

    tab_majority_decision = []
    
    for id, values in dataset.groupby('id_original'):

        line = None #reset line

        if not n_annotators: #if no amount of min annotators is provided
            line = make_line(values, equality)
        elif n_annotators and len(values.label)>=n_annotators: #if an amount of min annotators is provided and is respected
            line = make_line(values, equality)

        if line: #if a line was created
            tab_majority_decision.append(line)

    return pd.DataFrame(tab_majority_decision, columns=dataset.columns[3:10].tolist()+['label'])