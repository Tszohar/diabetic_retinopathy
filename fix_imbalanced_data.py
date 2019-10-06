from data import BDDataset
import pandas as pd
import parameters

if __name__ ==  '__main__':
    train_df = pd.read_csv(parameters.train_csv)
    count_class_0, count_class_1, count_class_2, count_class_3, count_class_4 = train_df.diagnosis.value_counts()

    # Split dataframe by classs
    df_class_0 = train_df[train_df['diagnosis'] == 0]
    df_class_1 = train_df[train_df['diagnosis'] == 1]
    df_class_2 = train_df[train_df['diagnosis'] == 2]
    df_class_3 = train_df[train_df['diagnosis'] == 3]
    df_class_4 = train_df[train_df['diagnosis'] == 4]

    # Over sampling classes 1,3,4 as class 2
    df_1_over = df_class_1.sample(count_class_1, replace=True)
    df_3_over = df_class_3.sample(count_class_1, replace=True)
    df_4_over = df_class_4.sample(count_class_1, replace=True)

    train_df_over = pd.concat([df_class_0, df_1_over, df_class_2, df_3_over, df_4_over], axis=0)
    #train_df_over.diagnosis.value_counts()
    train_df_over.to_csv(parameters.train_over_csv)