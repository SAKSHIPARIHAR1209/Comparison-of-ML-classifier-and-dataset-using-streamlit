from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
import streamlit as st
from sklearn import datasets
import pandas as pd
import numpy as np

st.title("Explore Different Classifier")

dataset_name=st.sidebar.selectbox("Select Datset",("Iris","Breast Cancer","Digits"))
# st.write(dataset_name)

classifier_name=st.sidebar.selectbox("Select Classifier",("KNN","SVM ","Decision Tree","Random Forest","GBDT"))

def get_dataset(dataset_name):
    if dataset_name=="Iris":
        data=datasets.load_iris()

    elif dataset_name=="Breast Cancer":
        data=datasets.load_breast_cancer()

    else:
        data=datasets.load_digits()
    X=data.data
    y=data.target

    return X,y

X,y=get_dataset(dataset_name)
st.write("Shape: ",X.shape)
st.write("No. of classes: ",len(np.unique(y)))

def add_parameter_ui(clf_name):

    params=dict()

    if clf_name=="KNN":
        K=st.sidebar.slider("K",1,15)
        params["K"]=K

    elif clf_name=="SVM":
        C=st.sidebar.slider("C",0.01,10.0)
        random_state=st.sidebar.slider("random_state",0,100)
        params["C"]=C
        params["random_state"]=random_state

    elif clf_name=="Decision Tree":
        max_leaf_nodes=st.sidebar.slider("max_leaf_nodes",2,15)
        random_state=st.sidebar.slider("random_state",0,100)
        params["max_leaf_nodes"]=max_leaf_nodes
        params["random_state"]=random_state

    elif clf_name=="Random Forest":
        max_depth=st.sidebar.slider("max_depth",2,15)
        n_estimators=st.sidebar.slider("n_estimators",1,100)
        params["max_depth"]=max_depth
        params["n_estimators"]=n_estimators

    else:
        learning_rate=st.sidebar.slider("learning_rate",0.1,1.0)
        n_estimators=st.sidebar.slider("n_estimators",1,100)
        max_depth=st.sidebar.slider("max_depth",2,15)
        max_features=st.sidebar.slider("max_features",2,15)
        params["learning_rate"]=learning_rate
        params["n_estimators"]=n_estimators
        params["max_depth"]=max_depth
        params["max_features"]=max_features

    return params


params=add_parameter_ui(classifier_name)

def get_classifier(clf_name,params):
    if clf_name=="KNN":
        clf=KNeighborsClassifier(n_neighbors=params["K"])
        

    elif clf_name=="SVM":
        clf=SVC(C=params["C"],random_state=params["random_state"])
        

    elif clf_name=="Decision Tree":
        clf=DecisionTreeClassifier(max_leaf_nodes=params["max_leaf_nodes"],random_state=params["random_state"])
        
        
    elif clf_name=="Random Forest":
        clf=RandomForestClassifier(max_depth=params["max_depth"],n_estimators=params["n_estimators"])
        
        

    else:
        clf=GradientBoostingClassifier(learning_rate=params["learning_rate"],n_estimators=params["n_estimators"],max_depth=params["max_depth"],max_features=params["max_features"])
    
    return clf

clf=get_classifier(classifier_name,params)

#CLASSIFICATION

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)

accuracy=accuracy_score(y_test,y_pred)
st.write(f"Classifier={classifier_name}")
st.write(f"Accuracy={accuracy}")

#confusion matrix,roc
metrics = st.sidebar.multiselect("What metrics to plot?",('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))
def plot_metrics(metrics_list):
        st.set_option('deprecation.showPyplotGlobalUse', False)
        if 'Confusion Matrix' in metrics_list:
            st.subheader("Confusion Matrix") 
            plot_confusion_matrix(clf, X_test, y_test)
            st.pyplot()
        
        if 'ROC Curve' in metrics_list:
            st.subheader("ROC Curve") 
            plot_roc_curve(clf, X_test, y_test)
            st.pyplot()

        if 'Precision-Recall Curve' in metrics_list:
            st.subheader("Precision-Recall Curve")
            plot_precision_recall_curve(clf, X_test, y_test)
            st.pyplot()

plot_metrics(metrics)

# Applying PCA
pca_dr=st.sidebar.selectbox("Applying PCA :",("NO","YES"))
def pca_apply(pca_dr):
    if pca_dr=="YES":
        st.write("""
        # APPLYING PCA(Principal Component Analysis)
        """)

        pca = PCA(n_components=2)
        pca.fit(X_train)
        X_t_train = pca.transform(X_train)
        X_t_test = pca.transform(X_test)
        clasf =clf
        clasf.fit(X_t_train, y_train)
        acc=clasf.score(X_t_test, y_test)

        st.write(f"Classifier={classifier_name}")
        st.write(f"Accuracy with PCA={acc}")

        
        # PLOTTING

        X_fit=pca.fit_transform(X)

        x1=X_fit[:,0]
        x2=X_fit[:,1]

        plot=plt.figure()
        plt.scatter(x1,x2,c=y,alpha=0.8,cmap="viridis")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.colorbar()

        st.pyplot(plot)
pca_apply(pca_dr)
