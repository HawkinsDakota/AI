"""
This script creates, trains, and evaluates a linear SVM model to predict Ikea
furniture class using AlexNet output activations and 1000 Principle Components
of Histogram of Oriented Gradients features.

Author: Dakota Hawkins
CS640 Project
"""

import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import confusion_matrix, accuracy_score
from matplotlib import pyplot as plt
import seaborn as sns
import pickle as pkl
from sklearn.externals import joblib

import itertools

if __name__ == "__main__":
    with open('label_data.pkl', 'rb') as f:
        label_data = pkl.load(f)

    csv_file = 'projected.csv'
    projected_data = pd.read_csv(csv_file, header=0, index_col=0)
    label_w_no_none = label_data[0][0:274] + label_data[0][275:]
    projected_data['label'] = [label_data[0][i] for i in projected_data['label'].astype(int)]
    projected_data = projected_data.dropna()

    labels = projected_data['label']
    columns = projected_data.columns.values[1:]

    model = svm.SVC(kernel='linear', C=1.0)
    x_train, x_test, y_train, y_test = train_test_split(projected_data[columns],
                                                        labels)
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    y_test_int = [label_data[1][x] for x in y_test.values]
    pred_int = [label_data[1][x] for x in pred]
    cm = confusion_matrix(y_test, pred)
    with open('confusion_matrix.csv', 'wb') as f:
        np.savetxt(f, cm)


    denom = cm.sum(axis=1)[:, np.newaxis]
    denom[denom==0] = 1

    cm = cm.astype('float') / denom
    ax = sns.heatmap(cm, cmap='YlGnBu', xticklabels=False, yticklabels=False)
    plt.savefig('cm_matrix.png')
    plt.close()

    cm_class_order = list(set(y_test.values).union(pred))
    cm_class_order.sort()

    acc = accuracy_score(y_test, pred)
    class_acc = []
    for i in range(cm.shape[0]):
        class_acc.append(cm[i, i])

    class_acc = np.array(class_acc)

    ranked_labels = np.argsort(-1*class_acc)
    plt.xkcd()
    plt.plot(range(0, len(class_acc)), class_acc[ranked_labels])
    plt.title('Ranked Class Predictive Power')
    plt.xlabel('Class')
    plt.ylabel('Accuracy')
    plt.show()

    label_ranks = {"Labels": np.array(cm_class_order)[ranked_labels],
                "Accuracy": class_acc[ranked_labels],
                "Rank": range(0, len(class_acc))}
    rank_df = pd.DataFrame(label_ranks)
    rank_df.to_csv('class_rank.csv')

    model = svm.SVC(kernel='linear', C=1.0, probability=True)
    model.fit(projected_data[columns], labels)
    joblib.dump(model, 'pca_svm.pkl')

