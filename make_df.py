import os
import pandas as pd


if __name__ == '__main__':
    dir_name = "CleanedText_Costum"
    data = {'file_name': [], 'words': []}
    for root, dirs, files in os.walk(dir_name, topdown=False):
        for file_name in files:
            with open(f'{dir_name}/{file_name}', 'r') as f:
                data['file_name'].append(file_name)
                data['words'].append(f.read().split())

    df = pd.DataFrame.from_dict(data)
    df.to_json("corpus.json")