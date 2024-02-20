# what-is-machine-learning

## _Description du projet_

Ce projet a pour but de nous faire réaliser une veille sur les différentes notions liées au Machine Learning (apprentissage automatique), afin d'aborder sereinement cette pierre angulaire de l'Intelligence Artificielle.  
Les notions sur lesquelles nous devons réaliser cette veille sont les suivantes:

- [ ] La science des données
- [ ] L’apprentissage automatique et/ou l’apprentissage profond
- [ ] L'apprentissage suprvisé
- [ ] L'apprentissage non-supervisé
- [ ] La classification supervisée
- [ ] La classification on-supervisée
- [ ] La régression
- [ ] La validation croisée
- [ ] Les données d’entraînement, les données de test et/ ou de validation
- [ ] Corrélation linéaire (de Pearson) entre deux variables
- [ ] Une fonction de coût
- [ ] La descente de gradient

Vous retrouverez dans la partie suivante les définitions accompagnées de quelques exemples pour chacune de ces notions.

## _Veille_

### **La science des données**

_Définition:_  
La science des données est l'étude des données afin d'en extraire des informations significatives. Il s'agit d'une approche pluridisciplinaire qui combine des principes et des pratiques issus des domaines des mathématiques, des statistiques, de l'intelligence artificielle et du génie informatique, en vue d'analyser de grands volumes de données. Cette analyse aide les scientifiques des données à poser des questions et à y répondre, comme Que s'est-il passé, Pourquoi cela s'est-il passé, Que va-t-il se passer et Que peut-on faire avec des résultats.

_A quoi sert la Science des données ?_  
La science des données sert à étudier les données de quatre principales manières :

1. Analyse descriptive
   **L'analyse descriptive** examine les données afin d'obtenir des informations sur ce qui s'est passé ou ce qui se passe dans l'environnement des données. Elle se caractérise par des visualisations de données telles que des diagrammes à secteurs, des histogrammes, des graphiques linéaires, des tableaux ou des récits générés. Par exemple, un service de réservation de vols peut enregistrer des données telles que le nombre de billets réservés chaque jour. L'analyse descriptive révélera alors les pics de réservation, les creux de réservation et les mois les plus performants pour ce service.

2. Analyse diagnostique
   **L'analyse diagnostique** est une plongée en profondeur ou un examen détaillé des données visant à comprendre pourquoi quelque chose s'est produit. Elle se caractérise par des techniques telles que l'analyse détaillée, la découverte de données, l'exploration de données et les corrélations. De multiples opérations et transformations de données peuvent être effectuées sur un jeu de données donné pour détecter des modèles uniques dans chacune de ces techniques. Par exemple, le service de vol peut analyser en détail un mois particulièrement performant pour mieux comprendre le pic de réservation. Par conséquent, il est possible de détecter que de nombreux clients se rendent dans une ville donnée pour assister à un événement sportif mensuel.

3. Analyse prédictive
   **L'analyse prédictive** utilise des données historiques pour faire des prévisions précises sur des modèles de données qui pourraient se présenter à l'avenir. Elle se caractérise par des techniques telles que le machine learning, la prédiction, la comparaison de modèles et la modélisation prédictive. Dans chacune de ces techniques, les ordinateurs sont formés à l'ingénierie inverse des liens de causalité dans les données. Par exemple, l'équipe du service des vols pourrait utiliser la science des données pour prédire les modèles de réservation de vols pour l'année suivante au début de chaque année. De même, le programme informatique ou l'algorithme peut analyser des données antérieures et prévoir des pics de réservation pour certaines destinations au mois de mai. Ayant anticipé les futurs besoins de voyage de ses clients, l'entreprise pourrait commencer à faire de la publicité ciblée pour ces villes à partir de février.

4. Analyse prescriptive
   **L'analytique prescriptif** permet de faire passer les données prédictives au niveau supérieur. Elle ne se contente pas de prédire ce qui risque de se produire, mais elle propose aussi une réponse optimale à ce résultat. Elle peut analyser les implications potentielles de différents choix et recommander la meilleure ligne de conduite. Elle utilise les analyses graphiques, la simulation, le traitement des événements complexes, les réseaux neuronaux et les moteurs de recommandation issus du machine learning.

Source: **Amazon Web Services**

_Illustration_  
![Illustration explicative de la Data Science](images/datasciences.png)

### **L'apprentissage automatique & l'apprentissage profond (Machine Learning & Deep Learning)**

_Définitions:_  
**Le Machine Learning** ou apprentissage automatique est un domaine scientifique, et plus particulièrement une sous-catégorie de l’intelligence artificielle. Elle consiste à laisser des algorithmes découvrir des « patterns », à savoir des motifs récurrents, dans les ensembles de données. Ces données peuvent être des chiffres, des mots, des images, des statistiques…

**Le Deep Learning** quant à lui est un sous-ensemble du Machine Learning où les réseaux neuronaux artificiels - des algorithmes conçus pour fonctionner comme le cerveau humain - apprennent à partir d'un grand nombre de données.


_Comment tout cela fonctionne t'il ?_  
Le fonctionnement du **Machine Learning** repose sur 4 étapes principales:

1. Sélectionner et préparer un ensemble de données d'entrainement qui seront utilisées pour nourrir le modèle de Machine Learning pour apprendre à résoudre des problèmes pour lequel il est conçu.  
   Les données peuvent être étiquettées (labelisées) pour indiquer au modèle les caractéristiques qu'il devra identifier. Si elles ne sont pas étiquettées le modèle devra repérer et extraire les caractéristiques récurrentes de lui même.

2. Sélectionner un algorithme a éxécuter sur l'ensemble des données d'entrainement. Le choix dépend du type et du volume des données d'entrainement et du type de problème à résoudre.

3.Entrainer l'algorithme. C'est un processus itératif. Des variables sont exécutées à travers l'algorithme et les résultats sont comparés avec ceux qu'il aurait pu produire. Les poids et le biais peuvent ensuite être ajustés pour accroître la précision du résultat. On exécute ensuite de nouveau les variables jusqu’à ce que l’algorithme produise le résultat correct la plupart du temps. L’algorithme, ainsi entraîné, est le modèle de Machine Learning.

4. Utiliser et améliorer le modèle. On utilise le modèle sur de nouvelles données, dont la provenance dépend du problème à résoudre.Par exemple, un modèle de Machine Learning conçu pour détecter les spams sera utilisé sur des emails.


Le **Deep Learning** est alimenté par des couches de réseaux neuronaux, qui sont des algorithmes vaguement modelés sur le fonctionnement du cerveau humain. L'entraînement avec de grandes quantités de données permet de configurer les neurones du réseau neuronal. Le résultat est un modèle de deep learning qui, une fois entraîné, traite de nouvelles données. Les modèles de deep learning recueillent des informations provenant de plusieurs sources de données et analysent ces données en temps réel, sans intervention humaine. Dans le cadre du deep learning, les processeurs graphiques (GPU) sont optimisés pour la formation de modèles, car ils peuvent traiter plusieurs calculs simultanément.

Sources: **datascientest.com & oracle.com**

_Illustration_  
![Illustration explicative du ML & DL](images/mldl.png)