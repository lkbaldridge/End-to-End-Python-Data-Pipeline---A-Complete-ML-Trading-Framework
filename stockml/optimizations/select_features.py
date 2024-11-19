import river.compat
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import RobustScaler, StandardScaler
import river
from mrmr import mrmr_classif
import pandas as pd
import numpy as np
from ..utils import config
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.model_selection import PredefinedSplit
from sklearn.metrics import make_scorer, precision_score
from typing import Any, Union

models = config.models


def select_features_l1(X_df: pd.DataFrame, y_df: pd.Series) -> list[str]:
    """
    Performs L1-regularized logistic regression feature selection with robust scaling.

    Implements feature selection using L1 regularization (Lasso) after robust scaling
    to handle outliers in financial data. This method is particularly effective for
    identifying the most important features while handling multicollinearity.

    Args:
        X_df (pd.DataFrame): Feature matrix containing predictor variables.
        y_df (pd.Series): Target variable series.

    Returns:
        list[str]: List of selected feature names that have non-zero coefficients
            in the L1-regularized model.

    Notes:
        - Uses RobustScaler to handle financial data outliers
        - Employs LogisticRegression with L1 penalty (Lasso)
        - C=1 sets the inverse of regularization strength
        - Uses liblinear solver for L1 optimization
    """

    scaler = RobustScaler()
    scaler.fit(X_df)

    X_scaled = scaler.transform(X_df)

    model = SelectFromModel(LogisticRegression(C=1, penalty='l1', solver='liblinear'))
    model.fit(X_scaled, y_df)

    selected_feat = X_df.columns[model.get_support()]
    
    return list(selected_feat)


def select_features_rf(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    max_depth: int = 10,
    class_weight: dict[int, float] = None,
    top_n: int = 15,
    n_estimators: int = 50,
    random_state: int = 0,
    importance_threshold: float = 0.05
) -> pd.DataFrame:
    """
    Performs Random Forest-based feature selection with importance ranking.

    Uses Random Forest classifier to rank features by importance, considering both
    feature importance scores and their stability across trees.

    Args:
        X_train (pd.DataFrame): Training feature matrix.
        y_train (pd.Series): Training target variable.
        max_depth (int, optional): Maximum depth of trees. Defaults to 10.
        class_weight (dict[int, float], optional): Class weights for imbalanced data.
            Defaults to None.
        top_n (int, optional): Number of top features to consider. Defaults to 15.
        n_estimators (int, optional): Number of trees in forest. Defaults to 50.
        random_state (int, optional): Random seed for reproducibility. Defaults to 0.
        importance_threshold (float, optional): Minimum importance score to keep feature.
            Defaults to 0.05.

    Returns:
        pd.DataFrame: Feature ranking DataFrame containing:
            - id: Rank position
            - indice: Original feature index
            - feature: Feature name
            - importances: Importance score

    Notes:
        - Utilizes all CPU cores (n_jobs=-1)
        - Calculates feature importance stability across trees
        - Prints detailed feature ranking information
    """
    
    model = RandomForestClassifier(n_estimators=n_estimators, 
                                   max_depth=max_depth,
                                   random_state=random_state, class_weight=class_weight,
                                   n_jobs=-1)
    
    model.fit(X_train, y_train)
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    feat_labels = X_train.columns
    
    std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0) 
    
    l1, l2, l3, l4 = [], [], [], []
    
    print("Feature ranking:")
    
    for f in range(X_train.shape[1]):
        if importances[indices[f]] > importance_threshold:
            print(f"{f + 1}. feature no:{indices[f]} feature name:{feat_labels[indices[f]]} ({importances[indices[f]]:.6f})")
        
        # Append details to lists
        l1.append(f + 1)
        l2.append(indices[f])
        l3.append(feat_labels[indices[f]])
        l4.append(importances[indices[f]])

    feature_rank = pd.DataFrame(zip(l1, l2, l3, l4), columns=['id', 'indice', 'feature', 'importances'])

    return feature_rank


def select_l1_rf_features(
    df_dict: dict[str, pd.DataFrame],
    y_df: pd.Series,
    cleaned_merged_df: pd.DataFrame,
    importance: float,
    test_size: int
) -> pd.DataFrame:
    """
    Performs two-stage feature selection combining L1 regularization and Random Forest.

    Implements a sophisticated feature selection pipeline that first uses L1 regularization
    to reduce dimensionality, followed by Random Forest importance scoring for final
    feature selection.

    Args:
        df_dict (dict[str, pd.DataFrame]): Dictionary of DataFrames containing different
            feature sets.
        y_df (pd.Series): Target variable series.
        cleaned_merged_df (pd.DataFrame): Complete dataset with all features.
        importance (float): Importance threshold for Random Forest selection.
        test_size (int): Number of samples to exclude from training.

    Returns:
        pd.DataFrame: Subset of cleaned_merged_df containing only selected features.

    Notes:
        - Two-stage selection process reduces overfitting
        - Removes duplicate features automatically
        - Prints progress information for each selection stage
    """

    print('Performing L1-based feature selection')
    features_dict_l1 = {}

    for key in list(df_dict.keys())[:-1]:
        feats = select_features_l1(df_dict[key][:-test_size], y_df[:-test_size])
        features_dict_l1[key] = feats
    
    print('L1 selection results')
    print('\n')

    for key in list(features_dict_l1.keys()):
        print(f'{key} - Number of features chosen by L1: {len(features_dict_l1[key])},
              '
              f'Number of starting features: {len(df_dict[key].columns.to_list())}')
        
    print('\n')
    print('Performing Random Forest-based feature selection on the L1-selected features')

    features_dict_rf = {}

    for key in list(features_dict_l1.keys()):
        feat_importances_df = select_features_rf(df_dict[key][features_dict_l1[key]][:-test_size], y_df[:-test_size])
        features_dict_rf[key] = feat_importances_df
    
    features_list = []

    for key in list(features_dict_l1.keys()):
        feature_df = features_dict_rf[key][features_dict_rf[key]['importances'] > importance]
        for feature in feature_df['feature'].to_list():
            features_list.append(feature)
    
    filtered_features = list(set(features_list))
    
    return cleaned_merged_df[filtered_features]


def select_mrmr_features(
    df_dict: dict[str, pd.DataFrame],
    y_df: pd.Series,
    cleaned_merged_df: pd.DataFrame,
    K: int,
    test_size: int
) -> pd.DataFrame:
    """
    Performs Minimum Redundancy Maximum Relevance (mRMR) feature selection.

    Implements mRMR feature selection across multiple feature sets to identify
    the most relevant and least redundant features for classification.

    Args:
        df_dict (dict[str, pd.DataFrame]): Dictionary of DataFrames containing different
            feature sets.
        y_df (pd.Series): Target variable series.
        cleaned_merged_df (pd.DataFrame): Complete dataset with all features.
        K (int): Number of features to select from each feature set.
        test_size (int): Number of samples to exclude from training.

    Returns:
        pd.DataFrame: Subset of cleaned_merged_df containing only selected features.

    Notes:
        - Selects features that maximize relevance to target
        - Minimizes redundancy among selected features
        - Combines features from multiple feature sets
        - Removes duplicate features automatically
    """

    selected_features = []
    
    for key in df_dict.keys():
        feat_importances = mrmr_classif(df_dict[key][:-test_size], y_df[:-test_size], K=K)
        selected_features.extend(feat_importances)
    
    filtered_features = list(set(selected_features))

    return cleaned_merged_df[filtered_features]


def run_sfs_with_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    test_size: int,
    model: Union[river.base.Estimator, sklearn.base.BaseEstimator],
    k_features: tuple[int, int] = (4, 8)
) -> list[str]:
    """
    Performs Sequential Feature Selection (SFS) with support for both River and sklearn models.

    Implements floating forward feature selection with predefined validation split,
    supporting both online learning (River) and batch learning (sklearn) models.

    Args:
        X_train (pd.DataFrame): Training feature matrix.
        y_train (pd.Series): Training target variable.
        test_size (int): Size of validation set for feature selection.
        model: Either River or sklearn model instance.
        k_features (tuple[int, int], optional): Range of features to select (min, max).
            Defaults to (4, 8).

    Returns:
        list[str]: Names of selected features that yield the best performance.

    Notes:
        - Automatically converts River models to sklearn format
        - Uses floating search for better feature subset optimization
        - Implements parallel processing for faster selection
        - Uses accuracy as the scoring metric
    """

    validation_indices = [-1] * (len(X_train) - test_size) + [0] * test_size
    piter = PredefinedSplit(validation_indices)
    
    if hasattr(model, 'learn_one'): 
        sklearn_model = river.compat.convert_river_to_sklearn(model)
    else:
        sklearn_model = model 
    
    sfs1 = SFS(sklearn_model,
               k_features=k_features,
               forward=True,
               floating=True,
               verbose=2,
               scoring = 'accuracy',
               cv=piter, 
               n_jobs=4)
    
    sfs1 = sfs1.fit(X_train, y_train)
    
    print(f"Best performing subset (features): {sfs1.k_feature_names_}")
    
    return list(sfs1.k_feature_names_)


