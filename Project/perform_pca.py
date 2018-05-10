"""
This script runs and saves Principle Component Analysis on Histogram of
Oriented Gradients (HOG) features.

Author: Dakota Hawkins
CS640 Project.
"""

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.externals import joblib

def pca_transform(data, pca):
    """
    Transform data to its principle components.

    Arguments:
        data (pandas.DataFrame): Dataframe containg HOG features to transform.
        pca (sklearn.decomposition.PCA): PCA model for HOG transformation.
    Returns:
        (pandas.DataFrame): transformed data.
    """
    pca_data = pca.transform(data)
    n_cols = pca_data.shape[1]
    pca_cols = ["PCA" + "0"*(len(str(n_cols)) - len(str(i + 1))) + str(i + 1)\
                for i in range(n_cols)]
    pca_df = pd.DataFrame(data=pca_data, index=data.index, columns=pca_cols)
    return pca_df

if __name__ == "__main__":

    csv_file = 'sub_img_data.csv'
    sub_img_data = pd.read_csv(csv_file, header=0, index_col=None)
    alex_columns = ['alex{}'.format(i) for i in range(1000)]
    hog_columns = ['hog{}'.format(i) for i in range(84672)]
    label_columns = ['label']
    hog_pca = PCA(n_components=1000)
    hog_pca.fit(sub_img_data[hog_columns])
    joblib.dump(hog_pca, 'pca.pkl')
    # hog_pca = joblib.load('pca.pkl')
    pca_df = pca_transform(sub_img_data[hog_columns], hog_pca)
    out = pd.concat([sub_img_data[['label'] + alex_columns], pca_df], 
                    axis=1)
    out.to_csv("projected.csv")
