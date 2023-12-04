import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':
    filepath = 'wineQualityRed.csv'
    df = pd.read_csv(filepath, sep=';')

    print('Dataframe info: ')
    df.info()

    print('\n\nDataframe stats: ')
    df.describe()

    print()
    for i in df.keys():
        print(f'{i} has {len(np.unique(df[i]))} unique values')

    print('\nVisualizing pair plot')
    cols2plot = ['fixed acidity', 'volatile acidity', 'citric acid', 'quality']
    sns.pairplot(df[cols2plot], kind='reg', hue='quality')
    plt.show()

    print('\nVisualizing Data')
    fig, ax = plt.subplots(1, figsize=(17, 10))
    ax = sns.boxplot(data=df)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    plt.show()

    # remove the outliers row in respect to total sulfuric dioxide
    df = df[df['total sulfur dioxide'] < 200]

    # calculate the z-sore ( other way to see outliers )
    # leave out the quality column
    cols2zscore = df.keys()
    cols2zscore = cols2zscore.drop('quality')
    for col in cols2zscore:
        mean = np.mean(df[col])
        sd = np.std(df[col], ddof=1)
        df[col] = (df[col] - mean) / sd

    print('Data stats after z-score calculation')
    df.describe()

    print('\nVisualizing Normalized Data')
    fig, ax = plt.subplots(1, figsize=(17, 10))
    ax = sns.boxplot(data=df)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    plt.show()
