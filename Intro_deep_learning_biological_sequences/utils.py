"""Python utilities for the notebooks"""

import collections
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import numpy as np
import pandas as pd
import pydotplus
import seaborn as sns

from io import StringIO
from IPython import display
from IPython.display import Image
from itertools import cycle
from operator import itemgetter
from scipy import interp
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import PolynomialFeatures  # To calculate the powers
                                                      # of our features
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


def make_summary_tables(res):
    """Turn a summary from statsmodel fitting results into two dataframes

    Parameters
    ----------
    res : [type]
        Summary from statsmodel fitting results

    Returns
    -------
    result_general_df : [type]
        contains general info and fit quality metrics
    result_fit_df : [type]
        coefficient values and confidence intervals
    """
    # transform second table to csv and read this as a dataFrame
    result_fit_df = pd.read_csv(StringIO(res.tables[1].as_csv() ),
                                         sep=",",
                                         index_col=0)
    result_fit_df.columns = [i.strip() for i in result_fit_df.columns]
    result_fit_df.index = [i.strip() for i in result_fit_df.index]

    # first table is trickier because the data is spread on to columns, and
    # there is title line.
    L = res.tables[0].as_html().split('\n')
    L.pop(1) # get rid of the title
    tmp = pd.read_html('\n'.join(L) , header=None)[0] # read as a dataframe,
                                                      # but with 4 columns.

    names = list(tmp[0]) + list(tmp[2])[:-2]  # columns 0 and 2 are metric names
    values = list(tmp[1]) + list(tmp[3])[:-2] # columns 1 and 3 are the
                                              # corresponding values.
                                              # NB : I exclude the last two
                                              # elements which are empty.

    result_general_df = pd.DataFrame({'Name': names , 'Value' : values},
                                     index=names,
                                     columns=['Value'] )

    return result_general_df, result_fit_df


def poly_fit(X,y):
    """[summary]

    Parameters
    ----------
    X : [type]
        [description]
    y : [type]
        [description]
    """
    poly = PolynomialFeatures(degree=3) # here we settle for a third degree
                                        # polynomial object
    X_poly = poly.fit_transform(X) # fit and transform the data
    print(X_poly[0,1])

    lr = LinearRegression()
    lr.fit(X_poly, y)
    y_predict=lr.predict(X_poly)
    R2 = r2_score(y, y_predict)
    MSE = mean_squared_error(y,y_predict)
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.plot(X[:,0], y, 'ko', label='Data')
    ax.plot(X[:,0], y_predict, 'r-.', label='Predicted')
    ax.legend(loc='best', fontsize=10)
    ax.set_title('R2={0:.2f}, MSE={1:.2f}'.format(R2,MSE), fontsize=13)
    ax.set_xlabel("Number of pedestrians per ha per min", fontsize=13)
    ax.set_ylabel("Breeding density(individuals per ha)", fontsize=13)
    # plt.show()

    print('fit param', lr.coef_[1:], lr.intercept_)

def poly_fit_train_test(X, y, seed, deg, ax=None):
    """Plot fit curves

    Parameters
    ----------
    X : [type]
        covariable matrix
    y : [type]
        dependent variable matrix
    seed : [type]
        random seed to determine train and test set
    deg : [type]
        degree of the polynomial to fit
    ax : [type], optional
        matplotlib ax to plot the fit, by default None

    Returns
    -------
    (float, float)
        R-squared on the train and the test set, respectively
    """

    poly = PolynomialFeatures(degree=deg) # here we settle for a third degree
                                          # polynomial object.
    X_poly=poly.fit_transform(X) # do the actual fit and transformation of data

    # we split X and y into a test set and train set
    # the train set will be used to fit
    # the test set will be used to evaluate the fit
    X_train, X_test, y_train, y_test = train_test_split(X_poly,
                                                        y,
                                                        random_state=seed,test_size=0.5)

    # print(X_poly)
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    # R2 with train set
    y_train_predict = lr.predict(X_train)
    R2_train = r2_score(y_train, y_train_predict)
    MSE_train = mean_squared_error(y_train, y_train_predict)

    # R2 with test set
    y_test_predict = lr.predict(X_test)
    R2 = r2_score(y_test, y_test_predict)
    MSE = mean_squared_error(y_test, y_test_predict)

    if not ax is None :

        # horrible code to sort the points
        y_predict = lr.predict(X_poly)
        xx, yy = zip(*sorted(
            [[u,v] for u,v in zip(X_poly[:,1],y_predict)], key=itemgetter(0)
            ))

        ax.plot(X_train[:,1], y_train, marker='o', linestyle='None',
                color='teal', label = 'train')
        ax.plot(X_test[:,1], y_test, marker='o', linestyle='None',
                color='orange', label='test')
        ax.plot(xx, yy,'r--', label='predicted')

        ax.set_title(
            'train : R2={0:.2f}, MSE={1:.2f}\n test : R2={2:.2f}, MSE={3:.2f}'.format(R2_train,MSE_train, R2,MSE), fontsize=13
            )

        ax.legend()

    return R2_train, R2


def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in.

    Parameters
    ----------
    x : [type]
        data to base x-axis meshgrid on
    y : [type]
        data to base y-axis meshgrid on
    h : float, optional
        stepsize for meshgrid, optional, by default .02

    Returns
    -------
    (ndarray, ndarray)
        The mesh
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax : [type]
        matplotlib axes object
    clf : [type]
        a classifier
    xx : ndarray
        x-axis of meshgrid
    yy : ndarray
        y-axis of meshgrid

    Returns
    -------
    [type]
        The output of `contourf`.

    Comments
    --------
    You can also optionally input as an argument a dictionary of params to pass to `contourf`.
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

def countour_lr_kypho(X, y, df, p='l2', c=10**8): #(number of nearest
                                                  # neighbors, feature matrix,
                                                  # label, voting rule)
    """[summary]

    Parameters
    ----------
    X : [type]
        [description]
    y : [type]
        [description]
    df : [type]
        [description]
    p : str, optional
        [description], by default 'l2'
    c : [type], optional
        [description], by default 10**8
    """

    models = LogisticRegression(penalty=p, C=c, class_weight='balanced')
    models = models.fit(X, y)

    # title for the plots
    titles = 'GLM Bernouilli'

    # Set-up 2x2 grid for plotting.
    fig, ax = plt.subplots(1, 1,figsize=(5, 5))
    # plt.subplots_adjust(wspace=0.4, hspace=0.4)

    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    y_pred_c = models.predict(X)

    plot_contours(ax, models, xx, yy, cmap=plt.cm.coolwarm, alpha=0.3)
    ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=40, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(titles+' accuracy= '+str(accuracy_score(y, y_pred_c)))
    ax.set_xlabel("age")
    ax.set_ylabel("number")
    plt.show()
    print([[w, list(df.columns)[i]] for i,w in enumerate(models.coef_[0])] +
          ['intercept',models.intercept_])

def countour_lr_kypho_train_test(df, y, seed, p='l2', c=10**8, plot=True):
    # (number of nearest neighbors, feature matrix, label, voting rule)
    """[summary]

    Parameters
    ----------
    df : [type]
        [description]
    y : [type]
        [description]
    seed : [type]
        [description]
    p : str, optional
        [description], by default 'l2'
    c : [type], optional
        [description], by default 10**8
    plot : bool, optional
        [description], by default True

    Returns
    -------
    [type]
        [description]
    """

    X_train, X_test, y_train, y_test = train_test_split(df,
                                                        y,
                                                        random_state=seed)
    scaler1 = StandardScaler()
    scaler1.fit(df)
    X_1=scaler1.transform(df)


    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train=scaler.transform(X_train)
    X_test=scaler.transform(X_test)
    models = LogisticRegression(penalty=p, C=c, class_weight='balanced',
                                solver='liblinear')
    models = models.fit(X_train, y_train)
    super_xx,super_yy=make_meshgrid(X_1[:, 0], X_1[:, 1])
    # title for the plots
    titles = 'GLM Bernoulli'
    y_pred_train_c = models.predict(X_train)
    y_pred_test_c = models.predict(X_test)
    if plot==True:
        # Set-up 2x2 grid for plotting.
        fig, ax = plt.subplots(1, 2,figsize=(14,7))
        # plt.subplots_adjust(wspace=0.4, hspace=0.4)

        X0, X1 = X_train[:, 0], X_train[:, 1]
        xx, yy = make_meshgrid(X0, X1)
        titles = 'GLM Bernoulli known'

        y_pred_train_c = models.predict(X_train)
        plot_contours(ax[0], models, super_xx, super_yy,
                      cmap=plt.cm.coolwarm, alpha=0.3)
        ax[0].scatter(X0, X1, c=y_train, cmap=plt.cm.coolwarm, s=40, edgecolors='k')
        ax[0].set_xlim(super_xx.min(), super_xx.max())
        ax[0].set_ylim(super_yy.min(), super_yy.max())
        ax[0].set_xticks(())
        ax[0].set_yticks(())
        ax[0].set_title(titles+' accuracy= '+str(accuracy_score(
            y_train,y_pred_train_c
            )))
        ax[0].set_xlabel("age")
        ax[0].set_ylabel("number")

        # y_pred_train_c=models.predict(X_train)
        # annot_kws = {"ha": 'center',"va": 'center'}
        # confusion_mc_c = confusion_matrix(y_train, y_pred_train_c)
        # df_cm_c = pd.DataFrame(confusion_mc_c,
        #                        index = ['Absent','Present'],
        #                        columns = ['Absent','Present'])


        # sns.heatmap(df_cm_c, annot=True,ax=ax[1,0], annot_kws=annot_kws)

        # ax[1,0].set_ylabel("True label")
        # ax[1,0].set_xlabel("Predicted label")

        titles = 'GLM Bernoulli new'
        X0, X1 = X_test[:, 0], X_test[:, 1]
        xx, yy = make_meshgrid(X0, X1)


        y_pred_test_c = models.predict(X_test)
        plot_contours(ax[1], models, super_xx, super_yy,
                      cmap=plt.cm.coolwarm, alpha=0.3)
        ax[1].scatter(X0, X1, c=y_test, cmap=plt.cm.coolwarm, s=40, edgecolors='k')
        ax[1].set_xlim(super_xx.min(), super_xx.max())
        ax[1].set_ylim(super_yy.min(), super_yy.max())
        ax[1].set_xticks(())
        ax[1].set_yticks(())
        ax[1].set_title(titles+' accuracy= '+str(accuracy_score(
            y_test, y_pred_test_c
            )))
        ax[1].set_xlabel("age")
        ax[1].set_ylabel("number")

        # confusion_mc_c2 = confusion_matrix(y_test, y_pred_test_c)
        # df_cm_c2 = pd.DataFrame(confusion_mc_c2,
        #                         index = ['Absent','Present'],
        #                         columns = ['Absent','Present'])


        # sns.heatmap(df_cm_c2, ax=ax[1,1], annot=True, annot_kws=annot_kws)
        # ax[1,1].set_ylabel("True label")
        # ax[1,1].set_xlabel("Predicted label")
        plt.tight_layout()
        plt.show()

        print([[w,list(df.columns)[i]] for i, w in enumerate(models.coef_[0])] +
              ['intercept',models.intercept_])


    return accuracy_score(y_train, y_pred_train_c), accuracy_score(y_test, y_pred_test_c)


def countour_lr2(p, X, y, c, mult):
    """[summary]

    Parameters
    ----------
    p : [type]
        [description]
    X : [type]
        [description]
    y : [type]
        [description]
    c : [type]
        [description]
    mult : [type]
        [description]
    """
    # Create the logistic regression object (with 3 main hyperparameters!!).
    # Penalty is either l1 or l2, C is how much weight we put on the
    # regularization, multi_class is how we proceed when multiclasses
    models = LogisticRegression(penalty=p, C=c, multi_class=mult)

    scaler=StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    models = models.fit(X, y)
    dico_color={0:'blue', 1:'white', 2:'red'}

    titles = 'Logistic regression penalty=' + str(p) + ' C=' + str(c) + '\n1./C=$\\alpha$=' + str(1./c)

    fig1, ax1 = plt.subplots(1,1,figsize=(10,5))
    # plt.subplots_adjust(wspace=0.4, hspace=0.4)
    # plt.subplot(1, 2, 1)
    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    plot_contours(ax1, models, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
    ax1.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    interc = models.intercept_
    wei = models.coef_

    for i in range(len(interc)):
        ax1.plot([xx.min(), xx.max()],
                 [-(interc[i] + wei[i][0] * xx.min()) / wei[i][1], -(interc[i] + wei[i][0] * xx.max()) / wei[i][1]],
                 color=dico_color[i],
                 ls='--')
    ax1.set_xlim(xx.min(), xx.max())
    ax1.set_ylim(yy.min(), yy.max())
    ax1.set_xticks(())
    ax1.set_yticks(())
    ax1.set_title(titles)

    # plt.savefig('C:\\Users\\sebas\\Desktop\\cours_scikit-learn\\Iris_example_knn_1_'+str(i)+'.pdf')
    # plt.subplots_adjust(wspace=0.4, hspace=0.4)
    # plt.subplot(1,2,1)
    # X0, X1 = X_test[:, 0],X_test[:, 1]
    # xx, yy = make_meshgrid(X0, X1)

    X0, X1 = X[:, 0], X[:, 1]

    xx = np.linspace(np.min(X0)-0.1, np.max(X0)+0.1, 100)
    yy = np.linspace(np.min(X1)-0.1, np.max(X1)+0.1, 100).T
    xx, yy = np.meshgrid(xx, yy)
    Xfull = np.c_[xx.ravel(), yy.ravel()]

    y_pred = models.predict(X)
    accuracy = accuracy_score(y, y_pred)
    # print("Accuracy (train) for %s: %0.1f%% " % (name, accuracy * 100))

    # View probabilities:
    probas = models.predict_proba(Xfull)
    n_classes = np.unique(y).size

    fig, ax = plt.subplots(1, n_classes, figsize=(10, 10 * n_classes))
    for k in range(n_classes):
        # ax.subplot(1, n_classes, k + 1)
        # plt.title("Class %d" % k)
        # print(k,min(probas[:, k]))

        if k == 0:
            ax[k].set_ylabel('LogiReg')

        imshow_handle = ax[k].imshow(probas[:, k].reshape((100, 100)),
                                     extent=(np.min(X0)-0.1,
                                             np.max(X0)+0.1,
                                             np.min(X1)-0.1,
                                             np.max(X1)+0.1),
                                     origin='lower',
                                     cmap='plasma')

        ax[k].set_xticks(())
        ax[k].set_xlim([np.min(X0)-0.1, np.max(X0)+0.1])
        ax[k].set_ylim([np.min(X1)-0.1, np.max(X1)+0.1])
        ax[k].set_yticks(())
        ax[k].set_title('Class ' + str(k))
        for i in range(len(interc)):
            ax[k].plot([np.min(X0) - 0.1, np.max(X0) + 0.1],
                       [-(interc[i] + wei[i][0] * (np.min(X0) - 0.1)) / wei[i][1], -(interc[i] + wei[i][0] * (np.max(X0) + 0.1)) / wei[i][1]],
                       color=dico_color[i],
                       ls='--')
        idx = (y_pred == k)

        if idx.any():
            ax[k].scatter(X[idx, 0],
                          X[idx, 1],
                          marker='o',
                          c=[dico_color[h] for h in y[idx]],
                          edgecolor='k')
        else:
            ax[k].set_visible(False)


    ax0 = plt.axes([0.15, 0.35, 0.7, 0.01])
    plt.title("Probability")
    plt.colorbar(imshow_handle, cax=ax0, orientation='horizontal')
    plt.show()

def countour_lr(p, X, y, c, mult):
    """[summary]

    Parameters
    ----------
    p : [type]
        [description]
    X : [type]
        [description]
    y : [type]
        [description]
    c : [type]
        [description]
    mult : [type]
        [description]
    """
    # Create the logistic regression object(with 3 main hyperparameters!!)
    # Penalty is either l1 or l2, C is how much weight we put on the
    # regularization, multi_class is how we proceed when multiclasses.
    models = LogisticRegression(penalty = p,C=c, multi_class=mult)
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        random_state=0,
                                                        stratify=y)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train=scaler.transform(X_train)
    X_test=scaler.transform(X_test)
    models = models.fit(X_train, y_train)
    dico_color = {0:'blue', 1:'white', 2:'red'}

    titles = 'Logistic regression penalty=' + str(p) + ' C=' + str(c) + '\n1./C=$\\alpha$=' + str(1./c)

    fig1, ax1 = plt.subplots(1, 2, figsize=(10, 5))
    # plt.subplots_adjust(wspace=0.4, hspace=0.4)
    # plt.subplot(1, 2, 1)
    X0, X1 = X_train[:, 0], X_train[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    plot_contours(ax1[0], models, xx, yy,cmap=plt.cm.coolwarm, alpha=0.8)
    ax1[0].scatter(X0, X1, c=y_train, cmap=plt.cm.coolwarm, s=20,
                   edgecolors='k')
    interc = models.intercept_
    wei = models.coef_

    for i in range(len(interc)):
        ax1[0].plot([xx.min(), xx.max()],
                    [-(interc[i] + wei[i][0] * xx.min())/wei[i][1],-(interc[i] + wei[i][0] * xx.max()) / wei[i][1]],
                    color=dico_color[i],
                    ls='--')
    ax1[0].set_xlim(xx.min(), xx.max())
    ax1[0].set_ylim(yy.min(), yy.max())
    ax1[0].set_xticks(())
    ax1[0].set_yticks(())
    ax1[0].set_title(titles)
    # plt.savefig('C:\\Users\\sebas\\Desktop\\cours_scikit-learn\\Iris_example_knn_1_'+str(i)+'.pdf')


    # plt.subplots_adjust(wspace=0.4, hspace=0.4)
    # plt.subplot(1,2,1)
    # X0, X1 = X_test[:, 0],X_test[:, 1]
    # xx, yy = make_meshgrid(X0, X1)

    plot_contours(ax1[1], models, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
    ax1[1].scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=plt.cm.coolwarm,
                   s=20, edgecolors='k')
    interc = models.intercept_
    wei = models.coef_
    for i in range(len(interc)):
        # print([-(interc[i]+wei[i][0]*xx.min())/wei[i][1],-(interc[i]+wei[i][0]*xx.max())/wei[i][1]])
        ax1[1].plot([xx.min(),xx.max()],
                    [-(interc[i] + wei[i][0] * xx.min()) / wei[i][1], -(interc[i] + wei[i][0] * xx.max()) / wei[i][1]],
                    color=dico_color[i],
                    ls='--')
    ax1[1].set_xlim(xx.min(), xx.max())
    ax1[1].set_ylim(yy.min(), yy.max())
    ax1[1].set_xticks(())
    ax1[1].set_yticks(())
    ax1[1].set_title(titles)

    plt.show()

    X = scaler.transform(X)

    X0, X1 = X[:, 0], X[:, 1]

    xx = np.linspace(np.min(X0)-0.1, np.max(X0)+0.1, 100)
    yy = np.linspace(np.min(X1)-0.1, np.max(X1)+0.1, 100).T
    xx, yy = np.meshgrid(xx, yy)
    Xfull = np.c_[xx.ravel(), yy.ravel()]

    y_pred = models.predict(X)
    accuracy = accuracy_score(y, y_pred)
    # print("Accuracy (train) for %s: %0.1f%% " % (name, accuracy * 100))

    # View probabilities:
    probas = models.predict_proba(Xfull)
    n_classes = np.unique(y).size

    fig, ax=plt.subplots(1, n_classes, figsize=(10, 10 * n_classes))
    for k in range(n_classes):
        # ax.subplot(1, n_classes, k + 1)
        # plt.title("Class %d" % k)
        # print(k,min(probas[:, k]))

        if k == 0:
            ax[k].set_ylabel('LogiReg')

        imshow_handle = ax[k].imshow(probas[:, k].reshape((100, 100)),
                                     extent=(np.min(X0)-0.1,
                                             np.max(X0)+0.1,
                                             np.min(X1)-0.1,
                                             np.max(X1)+0.1),
                                     origin='lower',
                                     cmap='plasma')

        ax[k].set_xticks(())
        ax[k].set_xlim([np.min(X0) - 0.1, np.max(X0) + 0.1])
        ax[k].set_ylim([np.min(X1) - 0.1, np.max(X1) + 0.1])
        ax[k].set_yticks(())
        ax[k].set_title('Class ' + str(k))
        for i in range(len(interc)):
            ax[k].plot([np.min(X0) - 0.1, np.max(X0) + 0.1],
                       [-(interc[i] + wei[i][0] * (np.min(X0) - 0.1)) / wei[i][1], -(interc[i] + wei[i][0] * (np.max(X0) + 0.1)) / wei[i][1]],
                       color=dico_color[i],
                       ls='--')
        idx = (y_pred == k)

        if idx.any():
            ax[k].scatter(X[idx, 0],
                          X[idx, 1],
                          marker='o',
                          c=[dico_color[h] for h in y[idx]],
                          edgecolor='k')
        else:
            ax[k].set_visible(False)


    ax0 = plt.axes([0.15, 0.35, 0.7, 0.01])
    plt.title("Probability")
    plt.colorbar(imshow_handle, cax=ax0, orientation='horizontal')
    plt.show()
    if n_classes>2:
        y = label_binarize(y, classes=np.arange(0, n_classes, 1))
        classifier = OneVsRestClassifier(LogisticRegression(penalty=p, C=c))
        y_score = classifier.fit(X_train, y_train).decision_function(X_test)

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(),
                                                  y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
        lw = 3
        # Plot all ROC curves
        plt.figure(figsize=(7,7))
        plt.plot(fpr["micro"], tpr["micro"],
                 label='micro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["micro"]),
                 color='deeppink', linestyle=':', linewidth=4)

        plt.plot(fpr["macro"], tpr["macro"],
                 label='macro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["macro"]),
                 color='navy', linestyle=':', linewidth=4)

        colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                     label='ROC curve of class {0} (area = {1:0.2f})'
                     ''.format(i, roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        # plt.title('Some extension of Receiver operating characteristic to multi-class')
        plt.title('Multi class Receiver operating characteristic curve')
        plt.legend(loc="lower right")
        plt.show()
    else:
        y_score_logi_r_c = models.decision_function(X_test)
        fpr_logi_r_c, tpr_logi_r_c, thre = roc_curve(y_test, y_score_logi_r_c)
        roc_auc_logi_r_c = auc(fpr_logi_r_c, tpr_logi_r_c)
        score = models.score(X,y)

        plt.figure()
        plt.xlim([-0.01, 1.00])
        plt.ylim([-0.01, 1.01])
        plt.plot(fpr_logi_r_c, tpr_logi_r_c, lw=3,
                 label='LogRegr ROC curve\n (area = {:0.2f})\n Acc={:1.3f}'.format(roc_auc_logi_r_c, score))
        plt.xlabel('False Positive Rate', fontsize=16)
        plt.ylabel('True Positive Rate', fontsize=16)
        plt.title('ROC curve (logistic classifier)', fontsize=16)
        plt.legend(loc='lower right', fontsize=13)
        plt.plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--')
        plt.axes().set_aspect('equal')
        plt.show()


def countour_SVM2(X, y, c, ker, deg, gam, mult):
    """[summary]

    Parameters
    ----------
    X : [type]
        [description]
    y : [type]
        [description]
    c : [type]
        [description]
    ker : [type]
        [description]
    deg : [type]
        [description]
    gam : [type]
        [description]
    mult : [type]
        [description]
    """
    # These are all the hyperparameters that are, in my opinion, important to
    # tune. `C` is again the good old inverse of the weight for l2
    # regularization, `kernel` is the dot product you want to use, `degree` is
    # the degree of the polynomial kernel you want to use, `gamma` is the
    # standard deviation for the Gaussian Radial Basis function,
    # `decision_function_shape` is used in case of multiclass, `proba = True`
    # is just here so we can draw the probability countour in our plot.
    models = svm.SVC(C=c, kernel=ker, degree=deg, gamma= gam, decision_function_shape=mult, probability=True)
    models = models.fit(X, y)
    dico_color={0:'blue', 1:'white', 2:'red'}

    titles = 'SVM' + ' C=' + str(c) + ' ' + ker

    fig1, ax1 = plt.subplots(1, 1)
    # plt.subplots_adjust(wspace=0.4, hspace=0.4)
    # plt.subplot(1, 2, 1)
    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    plot_contours(ax1, models, xx, yy,cmap=plt.cm.coolwarm, alpha=0.8)
    ax1.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    Z = np.asarray(models.decision_function(np.c_[xx.ravel(), yy.ravel()]))
    # print(np.shape(Z),Z.shape[0],print(np.shape(Z[:,0])))

    if ker=='linear':
        if len(set(y))==2:
            Zr = Z.reshape(xx.shape)
            ax1.contour(xx, yy, Zr, colors='k', levels=[-1, 0, 1], alpha=0.5,
                        linestyles=['--', '-', '--'])

    ax1.set_xlim(xx.min(), xx.max())
    ax1.set_ylim(yy.min(), yy.max())
    ax1.set_xticks(())
    ax1.set_yticks(())
    ax1.set_title(titles)
    # plt.savefig('C:\\Users\\sebas\\Desktop\\cours_scikit-learn\\Iris_example_knn_1_'+str(i)+'.pdf')

    plt.show()

    xx = np.linspace(np.min(X0)-5, np.max(X0)+5, 100)
    yy = np.linspace(np.min(X1)-5, np.max(X1)+5, 100).T
    xx, yy = np.meshgrid(xx, yy)
    Xfull = np.c_[xx.ravel(), yy.ravel()]

    y_pred = models.predict(X)
    accuracy = accuracy_score(y, y_pred)
    # print("Accuracy (train) for %s: %0.1f%% " % (name, accuracy * 100))

    # View probabilities:
    probas = models.predict_proba(Xfull)
    n_classes = np.unique(y_pred).size

    plt.figure(figsize=(10, 10 * n_classes))
    for k in range(n_classes):
        plt.subplot(1, n_classes, k + 1)
        # plt.title("Class %d" % k)
        if k == 0:
            plt.ylabel('SVM ' + ker)
        imshow_handle = plt.imshow(probas[:, k].reshape((100, 100)),
                                   extent=(np.min(X0) - 5,
                                           np.max(X0) + 5,
                                           np.min(X1) - 5,
                                           np.max(X1) + 5),
                                   origin='lower',
                                   cmap='plasma')
        plt.xticks(())
        plt.xlim([np.min(X0) - 5, np.max(X0) + 5])
        plt.ylim([np.min(X1) - 5, np.max(X1) + 5])
        plt.yticks(())
        plt.title('Class ' + str(k))


        idx = (y_pred == k)

        if idx.any():
            plt.scatter(X[idx, 0], X[idx, 1], marker='o',
                        c=[dico_color[h] for h in y[idx]], edgecolor='k')

    ax0 = plt.axes([0.15, 0.35, 0.7, 0.01])
    plt.title("Probability")
    plt.colorbar(imshow_handle, cax=ax0, orientation='horizontal')
    plt.show()


def countour_SVM(X, y, c, ker, deg, gam, mult):
    """[summary]

    Parameters
    ----------
    X : [type]
        [description]
    y : [type]
        [description]
    c : [type]
        [description]
    ker : [type]
        [description]
    deg : [type]
        [description]
    gam : [type]
        [description]
    mult : [type]
        [description]
    """
    # These are all the hyperparameters that are, in my opinion, important to
    # tune. `C` is again the good old inverse of the weight for l2
    # regularization, `kernel` is the dot product you want to use, `degree` is
    # the degree of the polynomial kernel you want to use, `gamma` is the
    # standard deviation for the Gaussian Radial Basis function,
    # `decision_function_shape` is used in case of multiclass, `proba = True`
    # is just here so we can draw the probability countour in our plot.
    models = svm.SVC(C=c, kernel=ker, degree=deg, gamma= gam, decision_function_shape=mult, probability=True)
    models = models.fit(X, y)
    dico_color={0:'blue', 1:'white', 2:'red'}

    titles = 'SVM' + ' C=' + str(c) + ' ' + ker

    fig1, ax1 = plt.subplots(1,1)
    # plt.subplots_adjust(wspace=0.4, hspace=0.4)
    # plt.subplot(1,2,1)
    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    plot_contours(ax1, models, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
    ax1.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    Z = np.asarray(models.decision_function(np.c_[xx.ravel(), yy.ravel()]))
    # print(np.shape(Z),Z.shape[0],print(np.shape(Z[:,0])))
    print(Z,np.shape(Z), type(Z))
    if ker == 'linear':
        if len(set(y)) == 2:
            Zr = Z.reshape(xx.shape)
            ax1.contour(xx, yy, Zr, colors='k', levels=[-1, 0, 1], alpha=0.5,
                        linestyles=['--', '-', '--'])

    ax1.set_xlim(xx.min(), xx.max())
    ax1.set_ylim(yy.min(), yy.max())
    ax1.set_xticks(())
    ax1.set_yticks(())
    ax1.set_title(titles)
    # plt.savefig('C:\\Users\\sebas\\Desktop\\cours_scikit-learn\\Iris_example_knn_1_'+str(i)+'.pdf')

    plt.show()

    xx = np.linspace(np.min(X0) - 5, np.max(X0) + 5, 100)
    yy = np.linspace(np.min(X1) - 5, np.max(X1) + 5, 100).T
    xx, yy = np.meshgrid(xx, yy)
    Xfull = np.c_[xx.ravel(), yy.ravel()]

    y_pred = models.predict(X)
    accuracy = accuracy_score(y, y_pred)
    # print("Accuracy (train) for %s: %0.1f%% " % (name, accuracy * 100))

    # View probabilities:
    probas = models.predict_proba(Xfull)
    n_classes = np.unique(y_pred).size

    plt.figure(figsize=(10,10*n_classes))
    for k in range(n_classes):
        plt.subplot(1, n_classes, k + 1)
        # plt.title("Class %d" % k)
        if k == 0:
            plt.ylabel('SVM ' + ker)
        imshow_handle = plt.imshow(probas[:, k].reshape((100, 100)),
                                   extent = (np.min(X0) - 5,
                                             np.max(X0) + 5,
                                             np.min(X1) - 5,
                                             np.max(X1) + 5),
                                   origin='lower',
                                   cmap='plasma')
        plt.xticks(())
        plt.xlim([np.min(X0) - 5, np.max(X0) + 5])
        plt.ylim([np.min(X1) - 5, np.max(X1) + 5])
        plt.yticks(())
        plt.title('Class ' + str(k))

        idx = (y_pred == k)

        if idx.any():
            plt.scatter(X[idx, 0], X[idx, 1], marker='o',
                        c=[dico_color[h] for h in y[idx]], edgecolor='k')

    ax0 = plt.axes([0.15, 0.35, 0.7, 0.01])
    plt.title("Probability")
    plt.colorbar(imshow_handle, cax=ax0, orientation='horizontal')

    plt.show()
    if n_classes>2:
        y = label_binarize(y, classes=np.arange(0,n_classes,1))
        classifier = OneVsRestClassifier(models)
        y_score = classifier.fit(X, y).decision_function(X)

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
        lw = 3

        # Plot all ROC curves
        plt.figure(figsize=(7,7))
        plt.plot(fpr["micro"], tpr["micro"],
                 label='micro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["micro"]),
                 color='deeppink', linestyle=':', linewidth=4)

        plt.plot(fpr["macro"], tpr["macro"],
                 label='macro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["macro"]),
                 color='navy', linestyle=':', linewidth=4)

        colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                     label='ROC curve of class {0} (area = {1:0.2f})'
                     ''.format(i, roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        # plt.title('Some extension of Receiver operating characteristic to multi-class')
        plt.title('Multi class Receiver operating characteristic curve')
        plt.legend(loc="lower right")
        plt.show()
    else:
        y_score_logi_r_c = models.decision_function(X)
        fpr_logi_r_c, tpr_logi_r_c, thre = roc_curve(y, y_score_logi_r_c)
        roc_auc_logi_r_c = auc(fpr_logi_r_c, tpr_logi_r_c)

        plt.figure()
        plt.xlim([-0.01, 1.00])
        plt.ylim([-0.01, 1.01])
        plt.plot(fpr_logi_r_c, tpr_logi_r_c, lw=3, label='SVM ROC curve\n (area = {:0.2f})'.format(roc_auc_logi_r_c))
        plt.xlabel('False Positive Rate', fontsize=16)
        plt.ylabel('True Positive Rate', fontsize=16)
        plt.title('ROC curve (logistic classifier)', fontsize=16)
        plt.legend(loc='lower right', fontsize=13)
        plt.plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--')
        plt.axes().set_aspect('equal')
        plt.show()


def countour_tree(X, y, crit, maxd, min_s, min_l, max_f):
    """[summary]

    Parameters
    ----------
    X : [type]
        [description]
    y : [type]
        [description]
    crit : [type]
        [description]
    maxd : [type]
        [description]
    min_s : [type]
        [description]
    min_l : [type]
        [description]
    max_f : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    # to understand what those hyperparameters stand for just check the first
    # example.
    models = DecisionTreeClassifier(criterion=crit,
                                    max_depth=maxd,
                                    min_samples_split=min_s,min_samples_leaf=min_l,
                                    max_features=max_f)
    models = models.fit(X, y)

    # title for the plots
    titles = 'Decision tree ' + ' '.join([str(crit),
                                          str(maxd),
                                          str(min_s),
                                          str(min_l),
                                          str(max_f)])

    # Set-up 2x2 grid for plotting.
    fig, ax = plt.subplots(1, 1)
    # plt.subplots_adjust(wspace=0.4, hspace=0.4)

    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    plot_contours(ax, models, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    # ax.set_xticks(())
    # ax.set_yticks(())
    ax.set_title(titles)
    # plt.savefig('C:\\Users\\sebas\\Desktop\\cours_scikit-learn\\Iris_example_knn_1_'+str(i)+'.pdf')
    plt.show()

    dot_data = tree.export_graphviz(models,
                                    feature_names=['x','y'],
                                    out_file=None,
                                    filled=True,
                                    rounded=True)
    graph = pydotplus.graph_from_dot_data(dot_data)

    colors = ('turquoise', 'orange')
    edges = collections.defaultdict(list)

    for edge in graph.get_edge_list():
        edges[edge.get_source()].append(int(edge.get_destination()))

    for edge in edges:
        edges[edge].sort()
        for i in range(2):
            dest = graph.get_node(str(edges[edge][i]))[0]
            dest.set_fillcolor(colors[i])

    return Image(graph.create_png())


def countour_RF(X, y, n_tree, crit, maxd, min_s, min_l, max_f):
    """[summary]

    Parameters
    ----------
    X : [type]
        [description]
    y : [type]
        [description]
    n_tree : [type]
        [description]
    crit : [type]
        [description]
    maxd : [type]
        [description]
    min_s : [type]
        [description]
    min_l : [type]
        [description]
    max_f : [type]
        [description]
    """
    models = RandomForestClassifier(n_tree,
                                    criterion=crit,
                                    max_depth=maxd,
                                    min_samples_split=min_s,min_samples_leaf=min_l,
                                    max_features=max_f)
    models = models.fit(X, y)
    dico_color={0:'blue', 1:'white', 2:'red'}
    # title for the plots
    titles = 'Random Forest ' + ' '.join([str(crit),
                                          str(maxd),
                                          str(min_s),
                                          str(min_l),
                                          str(max_f)])

    # Set-up 2x2 grid for plotting.
    fig, ax = plt.subplots(1, 1)
    # plt.subplots_adjust(wspace=0.4, hspace=0.4)

    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    Xfull = np.c_[xx.ravel(), yy.ravel()]

    plot_contours(ax, models, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_title(titles)
    plt.show()

    xx = np.linspace(np.min(X0) - 5, np.max(X0) + 5, 100)
    yy = np.linspace(np.min(X1) - 5, np.max(X1) + 5, 100).T
    xx, yy = np.meshgrid(xx, yy)
    Xfull = np.c_[xx.ravel(), yy.ravel()]

    y_pred = models.predict(X)
    accuracy = accuracy_score(y, y_pred)
    # View probabilities:
    probas = models.predict_proba(Xfull)
    n_classes = np.unique(y_pred).size

    plt.figure(figsize=(10, 10 * n_classes))
    for k in range(n_classes):
        plt.subplot(1, n_classes, k + 1)
        if k == 0:
            plt.ylabel('Random Forest')
        imshow_handle = plt.imshow(probas[:, k].reshape((100, 100)),
                                   extent=(np.min(X0) - 5,
                                           np.max(X0) + 5,
                                           np.min(X1) - 5,
                                           np.max(X1) + 5),
                                   origin='lower',
                                   cmap='plasma')
        plt.xticks(())
        plt.xlim([np.min(X0) - 5, np.max(X0) + 5])
        plt.ylim([np.min(X1) - 5, np.max(X1) + 5])
        plt.yticks(())
        plt.title('Class ' + str(k))

        idx = (y_pred == k)

        if idx.any():
            plt.scatter(X[idx, 0], X[idx, 1], marker='o',
                        c=[dico_color[h] for h in y[idx]], edgecolor='k')

    ax0 = plt.axes([0.15, 0.35, 0.7, 0.01])
    plt.title("Probability")
    plt.colorbar(imshow_handle, cax=ax0, orientation='horizontal')

    plt.show()

    models = DecisionTreeClassifier(criterion=crit,
                                    max_depth=maxd,
                                    min_samples_split=min_s,min_samples_leaf=min_l,
                                    max_features=max_f)
    models = models.fit(X, y)

        # title for the plots
    titles = 'Decision tree ' + ' '.join([str(crit),
                                          str(maxd),
                                          str(min_s),
                                          str(min_l),
                                          str(max_f)])

    # Set-up 2x2 grid for plotting.
    fig, ax = plt.subplots(1, 1)
    # plt.subplots_adjust(wspace=0.4, hspace=0.4)

    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    plot_contours(ax, models, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_title(titles)
    plt.show()

    xx = np.linspace(np.min(X0) - 5, np.max(X0) + 5, 100)
    yy = np.linspace(np.min(X1) - 5, np.max(X1) + 5, 100).T
    xx, yy = np.meshgrid(xx, yy)
    Xfull = np.c_[xx.ravel(), yy.ravel()]
    y_pred = models.predict(X)
    accuracy = accuracy_score(y, y_pred)
    # View probabilities:
    probas = models.predict_proba(Xfull)
    n_classes = np.unique(y_pred).size

    plt.figure(figsize=(10, 10 * n_classes))
    for k in range(n_classes):
        plt.subplot(1, n_classes, k + 1)
        if k == 0:
            plt.ylabel('Decision tree')
        imshow_handle = plt.imshow(probas[:, k].reshape((100, 100)),
                                   extent=(np.min(X0) - 5,
                                           np.max(X0) + 5,
                                           np.min(X1) - 5,
                                           np.max(X1) + 5),
                                   origin='lower',
                                   cmap='plasma')
        plt.xticks(())
        plt.xlim([np.min(X0) - 5, np.max(X0) + 5])
        plt.ylim([np.min(X1) - 5, np.max(X1) + 5])
        plt.yticks(())
        plt.title('Class ' + str(k))

        idx = (y_pred == k)

        if idx.any():
            plt.scatter(X[idx, 0], X[idx, 1], marker='o',
                        c=[dico_color[h] for h in y[idx]], edgecolor='k')

    ax0 = plt.axes([0.15, 0.35, 0.7, 0.01])
    plt.title("Probability")
    plt.colorbar(imshow_handle, cax=ax0, orientation='horizontal')

    plt.show()

class BlockingTimeSeriesSplit():
    """[summary]
    """
    def __init__(self, n_splits):
        self.n_splits = n_splits

    def get_n_splits(self, X, y, groups):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        k_fold_size = n_samples // self.n_splits
        indices = np.arange(n_samples)

        margin = 0
        for i in range(self.n_splits):
            start = i * k_fold_size
            stop = start + k_fold_size
            mid = int(0.8 * (stop - start)) + start
            yield indices[start: mid], indices[mid + margin: stop]


# This is not important. it is just to plot those graphs that will make things
# easier for you to understand. Just pay attention to the libraries involved
# and the two first lines of code.
def countour_lr_more(p, X, y, c, mult):
    """[summary]

    Parameters
    ----------
    p : [type]
        [description]
    X : [type]
        [description]
    y : [type]
        [description]
    c : [type]
        [description]
    mult : [type]
        [description]
    """
    # Create the logistic regression object(with 3 main hyperparameters!!)
    # Penalty is either l1 or l2, C is how much weight we put on the
    # regularization, multi_class is how we proceed when multiclasses.
    models = LogisticRegression(penalty=p, C=c, multi_class=mult)
    models = models.fit(X, y)
    dico_color={0:'blue', 1:'white', 2:'red'}

    titles = 'Logistic regression penalty=' + str(p) + ' C=' + str(c)

    fig1, ax1 = plt.subplots(1, 1)
    # plt.subplots_adjust(wspace=0.4, hspace=0.4)
    # plt.subplot(1,2,1)
    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    plot_contours(ax1, models, xx, yy,cmap=plt.cm.coolwarm, alpha=0.8)
    ax1.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    interc = models.intercept_
    wei = models.coef_
    for i in range(len(interc)):
        ax1.plot([xx.min(),xx.max()],
                 [-(interc[i] + wei[i][0] * xx.min()) / wei[i][1], -(interc[i] + wei[i][0] * xx.max()) / wei[i][1]],
                 color=dico_color[i],
                 ls='--')
    ax1.set_xlim(xx.min(), xx.max())
    ax1.set_ylim(yy.min(), yy.max())
    ax1.set_xticks(())
    ax1.set_yticks(())
    ax1.set_title(titles)
    # plt.savefig('C:\\Users\\sebas\\Desktop\\cours_scikit-learn\\Iris_example_knn_1_'+str(i)+'.pdf')

    plt.show()

    xx = np.linspace(np.min(X0) - 5, np.max(X0) + 5, 100)
    yy = np.linspace(np.min(X1) - 5, np.max(X1) + 5, 100).T
    xx, yy = np.meshgrid(xx, yy)
    Xfull = np.c_[xx.ravel(), yy.ravel()]

    y_pred = models.predict(X)
    accuracy = accuracy_score(y, y_pred)
    # print("Accuracy (train) for %s: %0.1f%% " % (name, accuracy * 100))

    # View probabilities:
    probas = models.predict_proba(Xfull)
    n_classes = np.unique(y_pred).size

    plt.figure(figsize=(10,10 * n_classes))
    for k in range(n_classes):
        plt.subplot(1, n_classes, k + 1)
        # plt.title("Class %d" % k)
        if k == 0:
            plt.ylabel('LogiReg')
        imshow_handle = plt.imshow(probas[:, k].reshape((100, 100)),
                                   extent=(np.min(X0)-5,
                                           np.max(X0)+5,
                                           np.min(X1)-5,
                                           np.max(X1)+5),
                                   origin='lower',
                                   cmap='plasma')
        plt.xticks(())
        plt.xlim([np.min(X0) - 5, np.max(X0) + 5])
        plt.ylim([np.min(X1) - 5, np.max(X1) + 5])
        plt.yticks(())
        plt.title('Class ' + str(k))
        for i in range(len(interc)):
            plt.plot([np.min(X0) - 5,np.max(X0) + 5],
                     [-(interc[i] + wei[i][0] * (np.min(X0) - 5)) / wei[i][1], -(interc[i] + wei[i][0] * (np.max(X0) + 5)) / wei[i][1]],
                     color=dico_color[i], ls='--')
        idx = (y_pred == k)

        if idx.any():

            plt.scatter(X[idx, 0], X[idx, 1], marker='o',
                        c=[dico_color[h] for h in y[idx]], edgecolor='k')

    ax = plt.axes([0.15, 0.45, 0.7, 0.01])
    plt.title("Probability")
    plt.colorbar(imshow_handle, cax=ax, orientation='horizontal')

    plt.show()
    if n_classes>2:
        y = label_binarize(y, classes=np.arange(0,n_classes,1))
        classifier = OneVsRestClassifier(LogisticRegression(penalty = p,C=c))
        y_score = classifier.fit(X, y).decision_function(X)

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
        lw = 3
        # Plot all ROC curves
        plt.figure(figsize=(7,7))
        plt.plot(fpr["micro"], tpr["micro"],
                 label='micro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["micro"]),
                 color='deeppink', linestyle=':', linewidth=4)

        plt.plot(fpr["macro"], tpr["macro"],
                 label='macro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["macro"]),
                 color='navy', linestyle=':', linewidth=4)

        colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                     label='ROC curve of class {0} (area = {1:0.2f})'
                     ''.format(i, roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        # plt.title('Some extension of Receiver operating characteristic to multi-class')
        plt.title('Multi class Receiver operating characteristic curve')
        plt.legend(loc="lower right")
        plt.show()
    else:
        y_score_logi_r_c = models.decision_function(X)
        fpr_logi_r_c, tpr_logi_r_c, thre = roc_curve(y, y_score_logi_r_c)
        roc_auc_logi_r_c = auc(fpr_logi_r_c, tpr_logi_r_c)
        score=models.score(X,y)

        plt.figure()
        plt.xlim([-0.01, 1.00])
        plt.ylim([-0.01, 1.01])
        plt.plot(fpr_logi_r_c, tpr_logi_r_c, lw=3, label='LogRegr ROC curve\n (area = {:0.2f})\n Acc={:1.3f}'.format(roc_auc_logi_r_c, score))
        plt.xlabel('False Positive Rate', fontsize=16)
        plt.ylabel('True Positive Rate', fontsize=16)
        plt.title('ROC curve (logistic classifier)', fontsize=16)
        plt.legend(loc='lower right', fontsize=13)
        plt.plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--')
        plt.axes().set_aspect('equal')
        plt.show()
