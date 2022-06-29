import numpy as np
import pandas
from pandas import read_csv
import matplotlib.pyplot as plt
from sklearn import tree, __all__, model_selection, preprocessing
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

path = 'C:\\Users\\giuse\\Downloads\\trainDdosLabelNumeric.csv'
pathTest = 'C:\\Users\\giuse\\Downloads\\testDdosLabelNumeric.csv'


def load(path):
    return read_csv(path)


def preElaborationData(data, cols):
    for c in cols:
        print(c)
        # determine statistics on c
        stat = (data[c].describe())
        print(stat)
    return stat


def removeColumns(dataframe, columns):
    removedColumns = []
    shape = dataframe.shape
    for c in columns:
        # I pass the attribute directly, taking the minimum and the maximum
        if dataframe[c].min() == dataframe[c].max():
            removedColumns.append(c)
    dataframe = dataframe.drop(columns=removedColumns)
    print('Removed columns: ', removedColumns)
    print('Dim before the removal: ', shape)
    print('Dim after the removal: ', dataframe.shape)
    return dataframe, removedColumns


def stratifiedKFold(X, y, folds, seed):
    # This cross - validation object is a variation of KFold that return the stratified folds.
    # The fold are made by preserving the percentage of samples for each class. Stratified because we must have a balanced partitioning
    # of dataset
    skf = model_selection.StratifiedKFold(n_splits=folds, random_state=seed, shuffle=True)
    '''
    If you work with shuffle = false, the dataset is divide by folds, simply following the order of the example.
    Shuffle = true, really randomly reorganize the dataset. Is important because without you risk that one fold contain examples of the same class.
    '''

    # empty lists declaration
    xTrainList = []
    xTestList = []
    yTrainList = []
    yTestList = []

    # looping over split, the output is formed by couple of test set and training set
    for trainIndex, testIndex in skf.split(X, y):
        print("TRAIN:", trainIndex, "TEST:", testIndex)
        xTrainList.append(X.iloc[trainIndex])
        xTestList.append(X.iloc[testIndex])
        yTrainList.append(y.iloc[trainIndex])
        yTestList.append(y.iloc[testIndex])
    return xTrainList, xTestList, yTrainList, yTestList


def decisionTreeLearner(X, y, criterion, ccp_alpha, seed):
    tree = DecisionTreeClassifier(criterion=criterion, random_state=seed, ccp_alpha=ccp_alpha)
    '''
    ccp_alpha is interesting for minimal cost complexity pruning, that is a pruning strategy implemented in SKlearn for decision tree. 
    Pruning strategy used to prune the decision tree. 
    '''

    '''
    Is important when we try to construct the tree we fix the random_state to seed to be sure that also executing 
    the algorithm several times, you construct the same tree. The randomic characterestic is a feature of this implementation
    not a feature of the algorithm.
    '''

    tree.fit(X, y)
    return tree


'''
For minimal cost-complexity pruning you need an alpha parameter specified by input. 
The minimal cost-complexity pruning could return different results on the same tree because the output will depend by alpha.
The algorithm is implemented to find the subtree of the original tree that minimize this metric. 
Any pruning activity will decrease one term of the formula and increase the others.
Start from the bottom I evalute each subtree found to detect if is better than the original tree. 
To prune the tree alpha should assume a particolare values.
I estimate for each subtree that i'm evaluating what is the value of alpha to achieve that the evaluation of the subtree is better than the original.
If the estimated value of alpha is greater than the value taken in input, means that is better to avoid the pruning.
'''


def showTree(T):
    plt.figure(figsize=(40, 10))
    plot_tree(T, filled=True, fontsize=10, proportion=True)
    n_nodes = T.tree_.node_count
    n_leaves = T.get_n_leaves()
    print('Number of nodes: ', n_nodes)
    print('Number of leaves: ', n_leaves)
    plt.show()


'''
This method allow us to compute the f1score as metric to evaluate the classification task.
Measuring the performance of the decision tree on the testing set passed as argument.
'''


def decisionTreeF1(YTest, XTest, tree):
    y_pred = tree.predict(XTest)
    score = f1_score(YTest, y_pred, average='weighted')
    return score


'''
This function perform some iteration to identify what is the best decision tree configuration in respect of the criterion and ccp_alpha.
For each parameter setup we have to compute on each trial of the cross-validation:
on the training the tree with configuration, compute the evaluation metrics on the five testing set and compute the average of the metrics.
At the end will return the configuration of decision tree that maximize the weighted f1score computed on the 5-folds cross validation.
'''


def determineDecisionTreekFoldConfiguration(xTrainList, xTestList, yTrainList, yTestList, seed):
    criterionList = ['entropy', 'gini']
    bestCcp_alpha = 0
    best_criterion = ''
    bestF1_score = 0
    minRange = 0
    maxRange = 0.05
    step = 0.001

    for i in np.arange(minRange, maxRange, step):
        for criterion in criterionList:
            f1 = []
            for x, y, z, w in zip(xTrainList, yTrainList, xTestList, yTestList):
                t = decisionTreeLearner(x, y, criterion, i, seed)
                f1.append(decisionTreeF1(w, z, t))
            avgF1 = np.mean(f1)
            print("average F1 score: ", avgF1)
            if avgF1 > bestF1_score:
                bestF1_score = avgF1
                best_criterion = criterion
                bestCcp_alpha = i
    return bestF1_score, bestCcp_alpha, best_criterion

    '''
    for x, y, z, w in zip(xTrainList, yTrainList, xTestList, yTestList):
        for i in np.arange(minRange, maxRange, step):
            for criterion in criterionList:
                t = decisionTreeLearner(x, y, criterion, i, seed)
                f1.append(decisionTreeF1(w, z, t))
                # score = decisionTreeF1(w, z, t)
                print('current f1 score: ', score)
                if score > bestF1_score:
                    bestF1_score = score
                    best_criterion = criterion
                    bestCcp_alpha = i
    return bestF1_score, bestCcp_alpha, best_criterion
    '''


ts = load(path)
ts_test = load(pathTest)
dim = ts.shape  # return a tuple representing the dimension of the DataFrame

print("The file loaded is the following: \n")
print(ts)

print("Matrix's size: \n")
print(dim)

print("The first five rows: \n")
print(ts.head())

print("Attributes labels: \n")
print(ts.columns)

# pre-elaboration
cols = list(ts.columns.values)  # list take an Index type from pandas library and return attribute labels as array
description = ts.describe()
description = ts_test.describe()

# remove the columns with same value on min-max (not only 0)
removeColumns(ts, cols)

"""
Plot the histogram of the class values
"""
plt.hist(ts['Label'], bins=len(cols))
plt.show()

# stratified K-fold CV
cols = list(ts.columns.values)  # retrieves all th
independentList = cols[0:ts.shape[
                             1] - 1]  # extract the list of columns with index between 0 and -1, excluding the last columns
print('Independent list: ', independentList)  # print all the dataset without the class columns
target = 'Label'
X = ts.loc[:, independentList]  # Projection of the original dataset on the independent attributes
y = ts[target]  # projection of the original dataset on the class
folds = 5
seed = 43
np.random.seed(seed)

'''
value to randomize the split, guarantee the same results generated in each iteration. 
It should be a number that control the sequence of random numbers that are generated. 
You have to set one seed and use the same seed in the project.
The important thing is that if I run the program twice, I should construct the same cross-validation of the original data
'''

ListXTrain, ListXTest, ListYTrain, ListYTest = stratifiedKFold(X, y, folds, seed)
'''
ListXTest is an array of 5 testing set projected on the independent variable, 
listYtest an array of 5 testing set projected on label,
listXTrain an array of 5 training set on label
'''

print(ListXTest)
print(ListXTrain)
print(ListYTest)
print(ListYTrain)

# DecisionTree
t = decisionTreeLearner(X, y, 'entropy', 0.001, seed)
showTree(t)

best_f1_score, best_ccp_alpha, best_criterion = determineDecisionTreekFoldConfiguration(ListXTrain, ListXTest,
                                                                                        ListYTrain, ListYTest, seed)

print('best F1 score is: ', best_f1_score)
print('best ccp_alpha: ', best_ccp_alpha)

new_tree = decisionTreeLearner(X, y, best_criterion, best_ccp_alpha, seed)
showTree(new_tree)

#
cols_test = list(ts_test.columns.values)  # retrieves all th
independentList = cols[0:ts_test.shape[1] - 1]
print(independentList)
target = 'Label'
X_test = ts_test.loc[:, independentList]
y_test = ts_test[target]
yPred = new_tree.predict(X_test)
score = decisionTreeF1(y_test, X_test, new_tree)
print(score)

'''
La diagonale principale ti dice quanti sono gli examples correttamente predetti
Sull'asse x ho la classe effettivamente predetta dall'albero e sull'asse y quelle presenti nella colonna label del test set
Sulla diagonale si possono verificare gli errori sulle predizioni rispetto a quelle che sono le classi nel test set 
'''
cm = confusion_matrix(y_test, yPred, labels=new_tree.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=new_tree.classes_)
disp.plot()
plt.show()
print(classification_report(y_test, new_tree.predict(X_test), labels=[0, 1, 2, 3, 4]))

# kmeans
# 1step: remove the class (we already did that, the result is 'X')
# 2step: clustering without the class

'''
The class is used to compute the purity and evaluate quality of the clusters.
The goal is trying using clustering for network intrusion detection and comparing the clustering with classification.
The cluster is trained without class.
For the evaluation we use inertia that is the standard deviation of the distance of each examples from the centroids of clusters.
Better clustering = lower inertia.
We can use kmeans++ to select the seed not randomly the k, selecting k that will maximize the distance between the seeds
'''

train_cluster_scaled = MinMaxScaler()  # scaling the data to make it comparable in the same range and perform the evaluation
print(train_cluster_scaled.fit(X))
print('The scaled data are the following: ')
Xscaled = train_cluster_scaled.transform(X)
print(Xscaled, '\n')
test_cluster_scaled = preprocessing.minmax_scale(X_test)

inertia = []

K = range(2, 25)
for k in K:
    print("Computing clustering with number of cluster: ", k)
    km = KMeans(n_clusters=k)
    km = km.fit(Xscaled)
    inertia.append(km.inertia_)  # lower inertia values, better clustering
'''
Inertia allow us to understand the compactness of the clusters. 
Sum of squared distance will contain the value of the inertia for each cluster, in fact going on through the k
the value decrease. If the number decrease means that the examples inside the clusters are closer to the centroids.
'''

print('\nSum of squared distances:')
print(inertia)

plt.plot(K, inertia, 'bx-')  # the parameter bx- is useful to insert the legenda on the elbow graphic
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()

best_K = 10
km = KMeans(n_clusters=best_K, random_state=seed)
km = km.fit(Xscaled)


def assign_class_to_cluster(y, kmeans_labels):
    # kmeans_labels return all the labels to identify the clusters. Each examples that belong to the corresponding
    # clusters.
    clusters = set(kmeans_labels)  # create a set of labels removing the duplicates.
    classes = set(y)
    class_to_cluster = []
    N = 0
    purity = 0
    # loop through the clusters cardinality chosen before
    for c in clusters:
        # create an array of values, each values correspond to the c index of the element of cluster labels
        # starting from c = 0, will found all the values 0 in kmeans_labels storing the position of all the labels inside indices
        # Contain all the positions of c value looping over the clusters
        indices = [i for i in range(len(kmeans_labels)) if kmeans_labels[i] == c]
        # take the predictions on the c-th cluster
        # Same size of indices, containing the class associated to the clusters
        selected_classes = [y[i] for i in indices]
        max_class = 0
        max_predicted_class_frequency = 0  # pcf = predicted class frequency

        for cl in classes:
            # counting on the clusters the occurrance of the classes IN EACH CLUSTER
            pcf = selected_classes.count(cl)
            # counting the cardinality of clusters
            N = N + pcf
            if pcf > max_predicted_class_frequency:
                max_predicted_class_frequency = pcf
                max_class = cl
        # max_class is the class that has the majority to the cluster c
        purity = purity + max_predicted_class_frequency
        # the nearest class for each of the 25 clusters
        class_to_cluster.append(max_class)
    purity = purity / N
    return clusters, class_to_cluster, purity


clusters, class_to_cluster, purity = assign_class_to_cluster(y, kmeans_labels=km.labels_)
print("clusters: ", clusters)
print("class_to_cluster", class_to_cluster)
print("purity: ", purity)

clustering_predictions_test = km.predict(test_cluster_scaled)
print(clustering_predictions_test)
predictions_test = [class_to_cluster[c] for c in clustering_predictions_test]
print(predictions_test)


cf_mat = confusion_matrix(y_test, predictions_test)
disp = ConfusionMatrixDisplay(confusion_matrix=cf_mat)
disp.plot()
plt.show()
print(classification_report(y_test, predictions_test))

