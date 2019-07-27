import os
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

data_labels = '/home/guy/tsofit/blindness detection/train_original.csv'


if __name__ == '__main__':

    src_df = pd.read_csv(data_labels)
    src_df.sort_values(by=['diagnosis'])
    train_df = pd.DataFrame(columns=['id_code', 'diagnosis'])
    validation_df = pd.DataFrame(columns=['id_code', 'diagnosis'])
    for level in range(5):
        level_df = src_df[src_df['diagnosis'] == level]
        train_size = int(0.80 * len(level_df))
        shuffled_indx = np.random.permutation(len(level_df))

        train_df = train_df.append(level_df.iloc[shuffled_indx[:train_size]])
        validation_df = validation_df.append(level_df.iloc[shuffled_indx[train_size:]])
    train_df = shuffle(train_df)
    validation_df = shuffle(validation_df)
    train_df.to_csv(os.path.join(os.path.dirname(data_labels), 'train.csv'))
    validation_df.to_csv(os.path.join(os.path.dirname(data_labels), 'validation.csv'))


