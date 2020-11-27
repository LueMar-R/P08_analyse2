#################################
#   DEPLOIEMENT ENVIRONNEMENT   #
#################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sc
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns

#################################
#    CHARGEMENT DES FICHIERS    #
#################################

creditcsv = pd.read_csv('Data_Vis/credit_immo.csv',sep=",")
creditxls = pd.read_excel('Data_Vis/credit_immo.xls')
creditjson = pd.read_json('Data_Vis/credit_immo.json')

#################################
#        2. MANIPULATION        #
#################################

#   CREATION DATAFRAME (BDD)    #
	
	# 6 lignes (remplissage random), 4 colonnes représentant les variables "taux_de_ventes, croissance_vente, ratio_benefice, ratio_perte".
a = np.random.randn(6,4) # rand : insertion de valeurs positives uniquement | randn : valeurs + et -
data = pd.DataFrame(data=a, columns=['taux_de_ventes', 'croissance_vente', 'ratio_benefice', 'ratio_perte'], index=[0, 1, 3, 5, 6, 7])


#       DONNEES MANQUANTES      #

data = data.reindex([0, 1, 2, 3, 4, 5]) # introduction de données manquantes en lignes 2 et 4

print(data.isnull()) # imprime "True" lorsqu'il rencontre une valeur manquante 

a = data.isnull().sum() # retourne le nombre de valeurs NaN par colonne
print("NaN par colonnes : \n",a)
a = data.isnull().sum().sum()  # nombre total de valeurs manquantes dans la dataframe
print("Total de NaN dans la DataFrame : \n",a) # 8
lignes_nan = data.index[data.isnull().any(axis=1)] # liste des lignes avec des données manquantes
print("lignes avec des Nan : \n",lignes_nan)

a = data.isnull()
d2 = data[a.any(axis=1)]
print(d2)

df = data.fillna(0) # remplacement des NaN par 0
print(df.isnull())

data.dropna(inplace=True) # suppression des lignes contenant des valeurs manquantes
print(data)

#################################
#   3. TRAITEMENT DES DONNEES   #
#################################

#   TRAITER VALEURS MANQUANTES  #
#          SIMPLEIMPUTER        #

print(creditcsv.shape) # (19, 10)
print(creditcsv.head(3))
	#        ID_NOM  Niv_Etude_Bac  age contrat_de_travail  ...  etat_civile  apport enfant_a_Charge  Solvable
	# 0      jean            3.0   45                CDI  ...            M     0.3             3.0       OUI
	# 1   VANESSA            5.0   28                CDI  ...            M     0.1             0.0       OUI
	# 2  TARCISSE            0.0   55                CDI  ...            C     0.0             0.0       NON

print(creditcsv.isnull().sum()) # on voit qu'il y a des NaN dans cretaines colonnes (Niv_Etude_Bac et enfant_a_Charge)

	# La solvabilité est ce qu'on cherche à prédire : on affecte donc ses valeurs à la sortie y :
y = creditcsv.iloc[:,9].values
print("y=",y) # y= ['OUI' 'OUI' 'NON' 'OUI' 'NON' 'NON' 'NON' 'NON' 'OUI' 'OUI' 'OUI' 'NON' 'NON' 'NON' 'NON' 'NON' 'OUI' 'OUI' 'OUI']
	# Le reste des données sont des entrées. On ne souhaite aps prendre en compte les noms dans le jeu de données.
X = creditcsv.iloc[:,1:9].values
print(X[:3]) # X est une matrice (et non plus un dataframe). si on imprime les 3 premières lignes de X :
	# [[3.0 45 'CDI' 40000 4000 'M' 0.3 3.0]
	#  [5.0 28 'CDI' 30500 1320 'M' 0.1 0.0]
	#  [0.0 55 'CDI' 28000 40000 'C' 0.0 0.0]]

imputer = SimpleImputer(missing_values=np.nan, strategy='mean') # On crée un Imputeur
X[:,7:] = imputer.fit_transform(X[:,7:]) # Transformation des valeurs manquantes en moyenne : 1.333
X[:,:1] = imputer.fit_transform(X[:,:1]) # Transformation des valeurs manquantes en moyenne : 1.444

	# equivaut à :
	# imputer.fit(X[:,0:1]) # on l'adapte aux données
	# X[:,7:] = imputer.transform(X[:,7:]) # Transformation des valeurs manquantes en moyenne
	# imputer.fit(X[:,0:1])
	# X[:,:1] = imputer.transform(X[:,:1])
print(X)

#  ENCODER VAL. CATEGORIELLES   #
#         LABELENCODER          #

	# Les colonnes catégorielles sont : contrat_de_travail, etat_civile (données qualitatives, col 2. et 5.)
encoder = LabelEncoder()
X[:,2] = encoder.fit_transform(X[:,2])
X[:,5] = encoder.fit_transform(X[:,5])
print(X)

# FRACTIONNEMENT - ENTAINEMENT  #
#     TRAINING AND TEST SET     #

	# on veut qu'une partie des données serve à l'apprentissage (_train) et l'autre aux tests de validation (_test)
	# il faut donc séparer l'échantillon en deux parties. En général on garde 20% des données pour les test, 80% pour l'entrainement
X_train, X_test, y_train_, y_test = train_test_split(X,y, test_size = 0.2)
print("train set :", X_train.shape) # (15, 8)
print("test set :", X_test.shape) # (4, 8)

# MISE A L'ECHELLE DES FEATURES #
#         STANDARDSCALER        #

scaler = StandardScaler()
X_test = scaler.fit_transform(X_test)
print(X_test)

#################################
#  4. VISUALISATION DE DONNEES  #
#################################

#     CHARGEMENT DU FICHIER     #
df = pd.read_csv("Data_Vis/Montant_Temps.csv", sep=",")
print(df.head(3))
	#    Montant_du_capitale  Temps
	# 0                    1      5
	# 1                   12      6
	# 2                   13      7

#     DECOUPAGE DES DONNEES     #
x = df.iloc[:,1].values
y = df.iloc[:,0].values

#    VISUALISATION GRAPHIQUE    #
plt.plot(x,y, color = "green", linestyle='dashed')
plt.xlabel('temps')
plt.ylabel('Montant du capital')
#plt.show()

plt.scatter(x,y)
plt.xlabel('temps')
plt.ylabel('Montant du capital')
#plt.show()


#################################
#     5. ANALYSE DE DONNEES     #
#################################

#  MESURE DE TENDANCE CENTRALE  #
data = pd.read_csv("Data_Vis/tendance_centrale.csv", sep=",")
print(data.head(3))
	#         Nom  Age  Rating
	# 0   ANATOLE   55    4.53
	# 1  Vincent    59    3.54
	# 2   Brenda    55    3.98

print("Age :\n",
	"moyenne :", data['Age'].mean(), "\n",
	"mediane :", data['Age'].median(), "\n",
	"mode :", data['Age'].mode())

print("Rating :\n"
	"moyenne :", data['Rating'].mean(), "\n",
	"mediane :", data['Rating'].median(), "\n",
	"mode :", data['Rating'].mode()) # ici il y a plusieurs résultats pour le mode.

#     ANALYSE DE LA VARIANCE    #

print("Age - var:\n",data['Age'].var(ddof=0))
print("Age - ect:\n",data['Age'].std(ddof=0))
print("Rating - var:\n",data['Rating'].var(ddof=0))
print("Rating - ect:\n",data['Rating'].std(ddof=0))

	# boites à moustaches
age = data.iloc[:,1].values
rating = data.iloc[:,2].values

fig1, ax1 = plt.subplots()
ax1.set_title('Boîte à moustaches - âge')
ax1.boxplot(age, vert =False)
#plt.show()

fig2, ax2 = plt.subplots()
ax2.set_title('Boîte à moustaches - rating')
ax2.boxplot(rating, vert =False)
#plt.show()

#     ANALYSE DE CORRELATION    #

df = pd.read_csv("Data_Vis/iris.csv", sep=",")
print(df.head(3))
	#    longueur_sepal  largeur_sepal  longueur_petal  largeur_petal  espèce
	# 0             5.1            3.5             1.4            0.2  setosa
	# 1             4.9            3.0             1.4            0.2  setosa
	# 2             4.7            3.2             1.3            0.2  setosa

print(df.shape) # (150, 5)

	# On souhaite déterminer s'il existe une relation les variables : les différentes dimensions petale/sepale
print(df.corr())

g = sns.PairGrid(df)
g.map(sns.scatterplot)
plt.show()
