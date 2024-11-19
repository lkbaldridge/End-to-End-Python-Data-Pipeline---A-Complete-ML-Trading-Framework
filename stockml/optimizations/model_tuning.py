import sklearn.feature_selection as fs
import sklearn.ensemble as ens
import sklearn.gaussian_process as gp
from sklearn.metrics import precision_score, confusion_matrix
import river
import xgboost as xgb
import flaml.default as fd
import river
from river import preprocessing, tree, metrics, evaluate, ensemble, stream, compat
import optuna
from optuna.trial import Trial
import numpy as np
import pandas as pd
from copy import copy
from river.forest import ARFClassifier
from ..utils.config import get_model_params
from ..dataset.prepare import create_datastream
from typing import Any

def extract_value(y: Any) -> Any:
    """
    Extracts the first value from a dictionary or returns the input value if not a dictionary.

    Used primarily for handling River ML's stream iteration function which may be nested in dictionaries.

    Args:
        y (Any): Input value or dictionary to extract from.

    Returns:
        Any: First value from dictionary if input is dict, otherwise returns input unchanged.
    """
    if isinstance(y, dict):
        return next(iter(y.values()))
    return y


def optuna_classification_reward(
    X_data: pd.DataFrame, 
    y_data: pd.DataFrame, 
    model: river.base.Classifier,
    iterate: bool = True,
    n_trials: int = 250,
    directions: list[str] = ['maximize'],
    param_ranges: dict[str, Any] = None,
    exclude_params: list[str] = None
) -> optuna.study.Study:
    """
    Optimizes trading model hyperparameters using Optuna with custom financial reward metrics.

    Implements a sophisticated optimization process for trading models using online learning
    and custom financial metrics. Uses TPE (Tree-structured Parzen Estimators) sampling for
    efficient hyperparameter search.

    Args:
        X_data (pd.DataFrame): Feature matrix containing trading indicators.
        y_data (pd.DataFrame): Target variables including 'change_encoded' and 'diff_shift'.
        model (river.base.Classifier): Base River ML classifier to optimize.
        iterate (bool, optional): Whether to use online learning. Defaults to True.
        n_trials (int, optional): Number of optimization trials. Defaults to 250.
        directions (list[str], optional): Optimization directions for each objective.
            Defaults to ['maximize'].
        param_ranges (dict[str, Any], optional): Custom parameter ranges for optimization.
            Defaults to None.
        exclude_params (list[str], optional): Parameters to exclude from optimization.
            Defaults to None.

    Returns:
        optuna.study.Study: Completed optimization study containing:
            - Trial history
            - Best parameters
            - Performance metrics
            - Trading statistics

    Notes:
        - Uses BaggingClassifier for ensemble predictions
        - Tracks precision, true positives, and financial metrics
        - Optimizes based on cumulative trading returns
        - Implements custom financial reward calculation
    """
    def objective(trial):
#       Suggest to use or not use each feature
#       selected_features = [feature for feature in features if trial.suggest_categorical(f'use_{feature}', [True, False])]

        base_model = get_model_params(trial, model, param_ranges= param_ranges, exclude_params=exclude_params)
        bagged = ensemble.BaggingClassifier(model=base_model, n_models=10)
        metric = metrics.Precision()
        cm = metrics.ConfusionMatrix()

        reward = 0
        losses = []
        profits = []

        if iterate:
        # Simulate online processing using stream.iter_array
            for i, (x, y) in enumerate(river.stream.iter_pandas(X_data, y_data['change_encoded'])):
                y_true = extract_value(y)
                y_pred = bagged.predict_one(x)
                bagged.learn_one(x, y_true)
                metric.update(y_true, y_pred)
                
                if y_pred == 1:
                # Update reward based on the value of y_data['diff_shift']
                    if y_true == -1:
                        losses.append(y_data['diff_shift'].iloc[i])
                    elif y_true == 1:
                        profits.append(y_data['diff_shift'].iloc[i])
                    elif y_true == 0:
                        if y_data['diff_shift'].iloc[i] < 0:
                            losses.append(y_data['diff_shift'].iloc[i])
                    
                    reward += y_data['diff_shift'].iloc[i]
                    
                cm.update(y_true, y_pred)
        
        else:
            pass

        true_pos = cm.true_positives(1)
        ave_loss = np.average(losses)
        ave_profit = np.average(profits)

#       trial.set_user_attr('trial_feats', features)
        trial.set_user_attr('true_pos', true_pos)
        trial.set_user_attr('precision', metric)
        trial.set_user_attr('average_losses', ave_loss)
        trial.set_user_attr('average_profits', ave_profit)

        print(f'Precision:{metric}')
        print(f'Positive Predictions:{true_pos}')
        print(f'Average Profit:{ave_profit}//Average Loss:{ave_loss}')

        return reward

    samplertpe = optuna.samplers.TPESampler(n_startup_trials=25, constant_liar=True, n_ei_candidates=15, multivariate=True, group=False)
    study = optuna.create_study(directions=directions, sampler=samplertpe)
    study.optimize(objective, n_trials=n_trials)

    return study


def show_optuna_results(
    study: optuna.study.Study,
    begin: int,
    end: int,
    sortby: int = 0
) -> pd.DataFrame:
    """
    Generates a detailed performance analysis DataFrame from an Optuna study's results.

    Extracts and formats optimization results including trading metrics, model parameters,
    and performance statistics. Allows for custom sorting and result filtering.

    Args:
        study (optuna.study.Study): Completed Optuna optimization study.
        begin (int): Starting index for trial selection.
        end (int): Ending index for trial selection.
        sortby (int, optional): Index of objective to sort by. Defaults to 0.

    Returns:
        pd.DataFrame: Results DataFrame containing:
            - Trial numbers and rewards
            - Precision metrics
            - Number of positive predictions
            - Average profits and losses
            - Model parameters (if available)
            All numeric values are formatted for readability.

    Notes:
        - Sorts trials by specified objective value
        - Formats financial metrics to 2-4 decimal places
        - Includes only completed trials
        - Preserves parameter configurations for each trial
    """

    completed_trials = [trial for trial in study.trials if (trial.values is not None)]
    sorted_trials = sorted(completed_trials, key=lambda trial: trial.values[sortby], reverse=True)

    trial_nums = []
    trial_rewards = []
    trial_precs = []
    trial_ave_profits = []
    trial_ave_losses = []
    trial_num_preds = []
    feats_list = []
    params_list = []

    for trial in sorted_trials[begin:end]:
        trial_feats = []
        trial_params = {}

        trial_nums.append(trial.number)
        trial_rewards.append(f'{trial.values[0]:.2f}')
        user_attrs = trial.user_attrs
        trial_precs.append(user_attrs['precision'])
        trial_num_preds.append(user_attrs['true_pos'])
        trial_ave_profits.append(f'{user_attrs["average_profits"]:.4f}')
        trial_ave_losses.append(f'{user_attrs["average_losses"]:.4f}')

        for i, j in trial.params.items():
            if trial.params.items():
                trial_params[i] = j

        if len(trial_feats) != 0:
            feats_list.append(trial_feats)
        if len(trial_params) != 0:
             params_list.append(trial_params)

    results_dict = {
                    'trial_num':trial_nums, 
                    'trial_rewards':trial_rewards, 
                    'trial_metric':trial_precs, 
                    'trial_num_preds':trial_num_preds, 
                    'trial_ave_profit':trial_ave_profits, 
                    'trial_ave_loss':trial_ave_losses
                    }

    if len(trial_params) != 0:
        results_dict['trial_params'] = params_list

    results_df = pd.DataFrame(results_dict)

    return results_df