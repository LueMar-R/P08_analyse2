#################################
#         1.DEPLOIEMENT         #
#################################

#   DEPLOIEMENT ENVIRONNEMENT   # 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stat
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns

#    CHARGEMENT DES FICHIERS    #

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
	# ID_NOM                0
	# Niv_Etude_Bac         1
	# age                   0
	# contrat_de_travail    0
	# Salaire               0
	# dette_anterieure      0
	# etat_civile           0
	# apport                0
	# enfant_a_Charge       1
	# Solvable              0
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
y = creditcsv.iloc[:,9].values # y est un vecteur (array)
	# Le reste des données sont des entrées. On ne souhaite pas prendre en compte ID_NOM dans le jeu de données.
X = creditcsv.iloc[:,1:9].values
print(X[:3]) # X est une matrice (et non plus un dataframe). Si on imprime les 3 premières lignes de X :
	# [[3.0 45 'CDI' 40000 4000 'M' 0.3 3.0]
	#  [5.0 28 'CDI' 30500 1320 'M' 0.1 0.0]
	#  [0.0 55 'CDI' 28000 40000 'C' 0.0 0.0]]

	# On sait qu'on a un Nan dnas la première colonne, et un dnas la dernière. On choisit de les remplacer par la moyenne
	# des autres valeurs (connues), afin de pouvoir conserver la ligne entière et traiter les autres données de cette ligne.
imputer = SimpleImputer(missing_values=np.nan, strategy='mean') # On crée un Imputeur
X[:,7:] = imputer.fit_transform(X[:,7:]) # Transformation des valeurs manquantes en moyenne : 1.333
X[:,:1] = imputer.fit_transform(X[:,:1]) # Transformation des valeurs manquantes en moyenne : 1.444
			# fit_transform equivaut à :
			# imputer.fit(X[:,0:1]) # on l'adapte aux données
			# X[:,7:] = imputer.transform(X[:,7:]) 
print(" après remplacement des valeurs manquantes :\n", X) # on voit que les (NaN) sont remplacées dans X par 1.333 et 1.444


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
	# il faut donc séparer l'échantillon en deux parties. 
	# En général on garde 20% des données pour les tests, 80% pour l'entrainement
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
plt.show()

plt.scatter(x,y)
plt.xlabel('temps')
plt.ylabel('Montant du capital')
plt.show()


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
plt.show()

fig2, ax2 = plt.subplots()
ax2.set_title('Boîte à moustaches - rating')
ax2.boxplot(rating, vert =False)
plt.show()

#     ANALYSE DE CORRELATION    #
#     RECHERCHE DE LIAISONS     #

df = pd.read_csv("Data_Vis/iris.csv", sep=",")
print(df.head(3))
	#    longueur_sepal  largeur_sepal  longueur_petal  largeur_petal  espèce
	# 0             5.1            3.5             1.4            0.2  setosa
	# 1             4.9            3.0             1.4            0.2  setosa
	# 2             4.7            3.2             1.3            0.2  setosa

print(df.shape) # (150, 5)

	# On souhaite déterminer s'il existe une relation les variables : les différentes dimensions petale/sepale
print(df.corr())
	#                 longueur_sepal  largeur_sepal  longueur_petal  largeur_petal
	# longueur_sepal        1.000000      -0.117570        0.871754       0.817941
	# largeur_sepal        -0.117570       1.000000       -0.428440      -0.366126
	# longueur_petal        0.871754      -0.428440        1.000000       0.962865
	# largeur_petal         0.817941      -0.366126        0.962865       1.000000

	# on voit que longueur_sepal et longueur_petal (87%) sont fortement correllées, 
	# longueur_sepal et largeur_petal (81%) également, 
	# ainsi que largeur_petal et longueur_petal (96%)
	# les indices étant positifs, ils évoluent dans le même sens (quand l'un augmente, l'autre augmente).
# g = sns.PairGrid(df)
# g.map(sns.scatterplot)

# h = df.corr(method='pearson').style.format("{:.2}").background_gradient(cmap=plt.get_cmap('coolwarm'))

# 	# ces résultats peuvent être affinés en prenant en compte l'espèce des iris. 
# g = sns.PairGrid(df, hue="espèce")
# g.map(sns.scatterplot) 
#plt.show()
	# ces nouveaux nuages de points montre qu'en réalité toutes les données longueur/largeur sépale/pétale
	# semblent corellées si l'on prend en compte l'espèce. La correlation est parfois forte, parfois plus nuancée.

#     ANALYSE DE CORRELATION    #
#        TEST DES LIAISONS      #

ls = df.iloc[:,0].values
ws = df.iloc[:,1].values
lp = df.iloc[:,2].values
wp = df.iloc[:,3].values

print("correlation de pearson entre largeur pétale et longueur sépale, toutes espèces confondues :")
print(stat.pearsonr(ls,wp)) # (0.8179411262715756, 2.325498079793272e-37)
	# on retrouve le coefficient de correlation de 0.82 du premier tableau. Ce résultat est accompagné d'une 
	# p-value qui permet d'évaluer si la liaison est statistiquement significative ou non.
	# Ici, la p-value (2.322e-37) est largement inférieure à 0.05, le résultat est donc significatif.
print("correlation de pearson entre longueur et largeur pétale, toutes espèces confondues :")
print(stat.pearsonr(lp,wp)) # (0.9628654314027963, 4.6750039073255585e-86)
	# à nouveau, la significativité du résultat du test de correlation est confirmée.
print("correlation de pearson entre longueur et largeur sépale, toutes espèces confondues :")
print(stat.pearsonr(ls,ws)) # (-0.11756978413300208, 0.15189826071144766)
	# par contre, comme on l'avait vu sur le premier tableau, il semble qu'il n'y ait aucun rapport entre la 
	# et largeur et la longueur du sépale.

	# CEPENDANT :
	# on a vu lors de l'analyse graphique que la variable "espèce" semble avoir une forte influence sur les
	# dimensions pétale / sépale, et que toutes les données semblent corellées si on les subdivise par espèce.

	# visualisation des différents types d'espèce :
print(df['espèce'].value_counts())
	# virginica     50
	# setosa        50
	# versicolor    50

	# on a donc trois espèces d'iris différentes. 50 échantillons ont été relevés pour chaque espèce, ce nombre
	# est assez élevé pour pouvoir analyser chacune des espèces individuellement.

	# création de sous-tableaux :

virginica = df[df['espèce'] == 'virginica']
setosa = df[df['espèce'] == 'setosa']
versicolor = df[df['espèce'] == 'versicolor']

	# premier test : corrélation longueur / largeur sépale
print("VIRGINICA - corrélation de Pearson entre longueur et largeur sépale :")
print(stat.pearsonr(virginica['longueur_sepal'], virginica['largeur_sepal'])) # (0.45722781639411314, 0.000843462472370877)
print("SETOSA - corrélation de Pearson entre longueur et largeur sépale :")
print(stat.pearsonr(setosa['longueur_sepal'], setosa['largeur_sepal']))# (0.7425466856651598, 6.709843017660477e-10)
print("VERSICOLOR - corrélation de Pearson entre longueur et largeur sépale :")
print(stat.pearsonr(versicolor['longueur_sepal'], versicolor['largeur_sepal'])) # (0.5259107172828248, 8.771860011973765e-05)
	# toutes espèces confondues, le coefficient de correlation de pearson trouvé était de -0.12 (donc aucune correlation 
	# entre longueur et largeur sepal). Finalement, losque l'on subdivise par espèce, on voit qu'au moins dans le cas du
	# SETOSA, il semble qu'il existe une relation entre ces deux dimensions de la fleur.

	# second test : corrélation longueur pétale / largeur sépale
print("VIRGINICA - corrélation de Pearson entre longueur pétale et largeur sépale :")
print(stat.pearsonr(virginica['longueur_petal'], virginica['largeur_sepal'])) # (0.40104457734278554, 0.0038977040683383264)
print("SETOSA - corrélation de Pearson entre longueur pétale et largeur sépale :")
print(stat.pearsonr(setosa['longueur_petal'], setosa['largeur_sepal']))# (0.17769996678227074, 0.2169789241968609)
print("VERSICOLOR - corrélation de Pearson entre longueur pétale et largeur sépale :")
print(stat.pearsonr(versicolor['longueur_petal'], versicolor['largeur_sepal']))# (0.5605220916929822, 2.3021679804283638e-05)
	# toutes espèces confondues, le coefficient de correlation trouvé était de -0.43. Ici, contrairement au premier test,
	# la corrélation évolue peu avec la subdivision par espèce. On ne peut pas conclure qu'il existe une relation entre
	# la largeur du sépale et la longueur du pétale chez les iris.

	# troisième test : corrélation longueur / largeur pétale
print("VIRGINICA - corrélation de Pearson entre longueur et largeur pétale :")
print(stat.pearsonr(virginica['longueur_petal'], virginica['largeur_petal'])) # (0.32210821590031835, 0.022535767279873883)
print("SETOSA - corrélation de Pearson entre longueur et largeur pétale :")
print(stat.pearsonr(setosa['longueur_petal'], setosa['largeur_petal']))# (0.3316300408041188, 0.018638915007537147)
print("VERSICOLOR - corrélation de Pearson entre longueur et largeur pétale :")
print(stat.pearsonr(versicolor['longueur_petal'], versicolor['largeur_petal']))# (0.7866680885228169, 1.27191570632364e-11)
	# le coefficient de correlation trouvé de 0.96 lors du test toutes espèces confondues laissait penser que ces deux 
	# dimensions sont très fortement corrélées. En réalité, à part dans le cas du VERSICOLOR, ces relations n'existent pas. 

	# revoyons les représentations graphiques des résultats pour le dernier test, par exemple :
plt.scatter(virginica['longueur_petal'], virginica['largeur_petal'], color = 'blue')
plt.title("virginica")
plt.xlabel('longueur_petal')
plt.ylabel('largeur_petal')
plt.show()

plt.scatter(setosa['longueur_petal'], setosa['largeur_petal'], color = 'orange')
plt.title("setosa")
plt.xlabel('longueur_petal')
plt.ylabel('largeur_petal')
plt.show()

plt.scatter(versicolor['longueur_petal'], versicolor['largeur_petal'], color = 'green')
plt.title("versicolor")
plt.xlabel('longueur_petal')
plt.ylabel('largeur_petal')
plt.show()
	# les deux premiers graphiques illustrent bien le fait qu'il n'y a pas de correlation entre longuer et largeur pétale 
	# pour les deux premières espèces. Pour le versicolor, on dicerne bien une tendance linéaire, mais avec des
	# écarts-types importants.

	# autre exemple :
plt.scatter(setosa['longueur_sepal'], setosa['largeur_sepal'], color = 'pink')
plt.title("setosa")
plt.xlabel('longueur_sepal')
plt.ylabel('largeur_sepal')
plt.show()
	# Pour la relation longueur / largeur sépale du setosa, on distingue également une tendance linéaire, mais là aussi  
	# avec des écarts-types importants.

