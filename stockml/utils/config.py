import river
import river.forest
import river.tree
import river.drift
import optuna
from optuna import trial, Trial

model_names = ['hoeffding_adaptive','arfc']

models = {model_names[0]:river.tree.HoeffdingAdaptiveTreeClassifier(), model_names[1]:river.forest.ARFClassifier()}

param_spaces = {
    'hoeffding_adaptive': {
        'grace_period': ('int', (50, 300)),
        'split_criterion': ('categorical', ['gini', 'info_gain', 'hellinger']),
        'delta': ('loguniform', (1e-7, 1e-1)),
        'tau': ('uniform', (0.01, 0.1)),
        'leaf_prediction': ('categorical', ['mc', 'nb', 'nba']),
        'nb_threshold': ('int', (0, 100)),
        'bootstrap_sampling': ('categorical', [True, False]),
        'drift_window_threshold': ('int', (10, 200)),
        'switch_significance': ('uniform', (0.01, 0.2)),
        'binary_split': ('categorical', [True, False]),
        'min_branch_fraction': ('uniform', (0.001, 0.1)),
        'max_share_to_split': ('uniform', (0.5, 1.0)),
        'remove_poor_attrs': ('categorical', [True, False]),
        'merit_preprune': ('categorical', [True, False])
    }
}



def get_drift_params(trial, drift_detector):
    if drift_detector == 'ADWIN':
        adwin_delta = trial.suggest_loguniform('adwin_delta', 1e-5, 1e-1)
        adwin_clock = trial.suggest_int('adwin_clock', 1, 64)  
        adwin_max_buckets = trial.suggest_int('adwin_max_buckets', 2, 10) 
        adwin_min_window_length = trial.suggest_int('adwin_min_window_length', 2, 20)  
        adwin_grace_period = trial.suggest_int('adwin_grace_period', 5, 50)  

        adwin_drift_detector = river.drift.ADWIN(
            delta=adwin_delta,
            clock=adwin_clock,
            max_buckets=adwin_max_buckets,
            min_window_length=adwin_min_window_length,
            grace_period=adwin_grace_period
        )

        return adwin_drift_detector
    
def remove_excluded_params(default_params, exclude_params):
    """
    Remove excluded parameters from the default parameters.

    Args:
        default_params (dict): The default parameter ranges and types.
        exclude_params (list): A list of parameter names to exclude.

    Returns:
        dict: Updated default parameters with excluded ones removed.
    """
    if exclude_params:
        for param_name in exclude_params:
            if param_name in default_params:
                del default_params[param_name]
    
    return default_params

    
def apply_custom_param_ranges(param_ranges, default_params):
    """
    Apply custom parameter ranges if provided.

    Args:
        param_ranges (dict): A dictionary of custom parameter ranges.
        default_params (dict): The default parameter ranges and types.

    Returns:
        dict: Updated parameters with custom ranges applied.
    """
    updated_params = default_params.copy()

    if param_ranges:
        for param_name in param_ranges:
            if param_name in default_params:
                updated_params[param_name] = param_ranges[param_name]
    
        return updated_params
    else:
        return default_params
    
def suggest_hyperparameter(trial, param_name, param_type, param_values):
    """
    Suggest a hyperparameter dynamically based on its type.

    Args:
        trial (optuna.Trial): The trial object from Optuna used for suggesting hyperparameters.
        param_name (str): The name of the parameter.
        param_type (str): The type of the parameter ('int', 'loguniform', 'uniform', 'categorical').
        param_values (tuple or list): The range or list of values for the parameter.

    Returns:
        The suggested value for the parameter.
    """
    if param_type == 'int':
        return trial.suggest_int(param_name, *param_values)
    elif param_type == 'loguniform':
        return trial.suggest_loguniform(param_name, *param_values)
    elif param_type == 'uniform':
        return trial.suggest_uniform(param_name, *param_values)
    elif param_type == 'categorical':
        return trial.suggest_categorical(param_name, param_values)
    else:
        raise ValueError(f"Unsupported parameter type: {param_type}")



def get_model_params(trial, model_name, param_ranges =  None, exclude_params = None):
    if model_name == 'hoeffding_adaptive':
        default_params = param_spaces[model_name]
        default_params = remove_excluded_params(default_params, exclude_params)
        default_params = apply_custom_param_ranges(param_ranges, default_params)
        trial_params_dict = {}


        for param_name, (param_type, param_values) in default_params.items():
            trial_params_dict[param_name] = suggest_hyperparameter(trial, param_name, param_type, param_values)

        trial_params_dict['drift_detector'] = get_drift_params(trial, 'ADWIN')

        return river.tree.HoeffdingAdaptiveTreeClassifier(**trial_params_dict, seed=42)
    
    elif model_name == 'arfc':
        pass