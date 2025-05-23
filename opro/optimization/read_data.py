import pandas as pd


def get_examples(self, path):
    df = pd.read_csv(path, sep=';', header=None)
    exs = df.reset_index().to_dict('records')
    exs = [{'text': x[1], 'label': int(x[2])} for x in exs]
    return exs

 