# Installation (fr)

## Python

* Python 3.7
* Pour Windows https://www.anaconda.com/distribution/
* Intaller `poetry`. Dans une invite à commande, normalement `conda install poetry`
  devrait suffire

## Paquet geo3dfeatures


* Récupérer le code source via Git

  `git clone https://git.oslandia.net/Oslandia-data/geo3dfeatures.git`

  Il y aura besoin du login et mot de passe (le même que pour se connecter à la
  plate-forme gitlab).

* Aller dans le dossier `geo3dfeatures` et lancer `poetry install`.

* Tester l'installation en exécutant `geo3d --help`. On devrait voir l'aide
  s'afficher comme ceci :

  ```
  usage: geo3d [-h] {info,sample,index,featurize,profile,cluster} ...

  Geo3dfeatures framework for 3D semantic analysis

  positional arguments: {info,sample,index,featurize,profile,cluster}
    info                Describe an input .las file
    sample              Extract a sample of a .las file
    index               Index a point cloud file and serialize it
    featurize           Extract the geometric feature associated to 3D points
    profile             Extract in a human-readable format the geometric
                        feature extraction time measurements
    cluster             Cluster a set of 3D points with a k-means algorithm

  optional arguments:
    -h, --help            show this help message and exit
  ```

## Répertoire de travail / data

Créer un dossier de "travail", i.e. où la donnée va vivre, en-dehors du dossier
source. Ce dossier doit avoir au moins cette structure :

- dossier `input`
- dossier `output`

Dans `input`, on y met les fichiers de nuages de points, e.g. `.las`. Pour tous les
fichiers générés par le programme, les données seront stockées dans `output`.

Pour le reste des étapes comme `index` ou extraire les *features* via `featurize`,
suivre le fichier [README.md](./README.md).
