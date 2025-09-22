import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

dataframe = pd.read_csv("synthetic.csv")
# print(dataframe.head())
dataframe.info()

# Visualisation des données 
sns.countplot(x = 'Class', data=dataframe)
plt.title('Visualisation de la répartition dans les classes')
# plt.show()

# Séparation des données en sous classes : entrainement et évaluation
# Attribution des données
X = dataframe[['Attr_A', 'Attr_B', 'Attr_C', 'Attr_D', 'Attr_E', 'Attr_F', 'Attr_G', 'Attr_H', 'Attr_I', 'Attr_J', 'Attr_K', 'Attr_L', 'Attr_M', 'Attr_N']]
y = dataframe['Class']
#  Séparation des données : 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 101)
clf = DecisionTreeClassifier(criterion='gini')

#Fit d'entraînement :
clf.fit(X_train, y_train)
print(clf.get_depth())

plt.figure(figsize=(15, 12))
tree.plot_tree(clf, rounded=True, filled=True)
plt.show()
# # Extraction des attributs
# features = list(dataframe.columns)
# print(features)

# # Récupération des classes
# classes = dataframe['Class'].unique()
# print(classes)
# print(dataframe.info())

# # Analyse des classes
# class_distribution = dataframe['Class'].value_counts()
# print(class_distribution)

# # class_stats = dataframe.groupby('Class').describe()


# # Visualisation de la séparabilité linéaire
# plt.figure(figsize=(8, 6))
# sns.scatterplot(x='Attr_J', y='Attr_I', hue='Class', data=dataframe, palette='Set1')
# plt.title('Visualisation de la séparabilité linéaire')
# plt.xlabel('Attr_J')
# plt.ylabel('Attr_I')
# plt.legend(title='Classe')
# plt.show()


# # SVM classification pour déterminer si nos données sont linéairement séparables


# X = dataframe.columns
# print(X.dtype)
# # Y = classes
# # clf = svm.SVC(decision_function_shape='ovo')
# # dec = clf.fit(X, Y)
# # dec.shape[1]