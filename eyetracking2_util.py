import numpy as np
import copy

import load_subjects as ls
import util
from itertools import izip_longest
import matplotlib.pyplot as plt
import eyetracking_hmm
import naive_eyetracking 

conditions = ['shrinky', 'noshrinky']

def plot_with_std_error_bars(x, ys, color_idx):
  if color_idx > 9:
    raise ValueError('Apparently the standard color wheel only supports single digit colors 0-9.' + \
                      'See https://matplotlib.org/users/colors.html')
  color_name = 'C' + str(color_idx)
  y_means = np.nanmean(ys, axis=1)
  y_stds =  np.nanstd(ys, axis=1)
  y_counts = np.logical_not(np.isnan(ys)).sum(axis=1)
  handle = plt.errorbar(x, y_means, y_stds/np.sqrt(y_counts), color=color_name, linewidth=1.0, capsize=4.0, capthick=1.0).lines[0]
  return handle

def plot_with_std_error_bands(x, ys, color_idx, CI_type='bootstrap'):
  if color_idx > 9:
    raise ValueError('Apparently the standard color wheel only supports single digit colors 0-9.' + \
                      'See https://matplotlib.org/users/colors.html')
  color_name = 'C' + str(color_idx)
  y_means = np.nanmean(ys, axis=1)
  handle, = plt.plot(x, y_means, color=color_name, linewidth=1.0)
  # plt.plot(x, y_means + y_stds/np.sqrt(y_counts), color=color_name, linewidth=0.5)
  # plt.plot(x, y_means - y_stds/np.sqrt(y_counts), color=color_name, linewidth=0.5)
  if CI_type == 'CLT': # Use central limit theorem to compute CIs
    y_stds =  np.nanstd(ys, axis=1)
    y_counts = np.logical_not(np.isnan(ys)).sum(axis=1)
    y_lower = y_means - y_stds/np.sqrt(y_counts)
    y_upper = y_means + y_stds/np.sqrt(y_counts)
  elif CI_type == 'bootstrap':
    ys_len, bootstrapped_sample_size = ys.shape
    num_bootstrap_repetitions = 2000
    ys_bootstrapped = np.zeros(ys.shape)
    y_means_bootstrapped = np.zeros((ys_len, num_bootstrap_repetitions))
    for bootstrap_rep in range(num_bootstrap_repetitions):
      sample_idxs = np.random.choice(bootstrapped_sample_size, size=bootstrapped_sample_size, replace=True)
      y_means_bootstrapped[:, bootstrap_rep] = np.nanmean(ys[:, sample_idxs], axis=1)
    y_lower = np.nanpercentile(y_means_bootstrapped, 2.5, axis=1)
    y_upper = np.nanpercentile(y_means_bootstrapped, 97.5, axis=1)
  else:
    raise ValueError('CI_type: ' + str(CI_type) + ' is not supported. Supported values are \'CLT\' and \'bootstrap\'.')
  plt.fill_between(x, y_lower, y_upper, where=y_upper>=y_lower, facecolor=color_name, interpolate=True, alpha=0.3)
  return handle

def reduce_to_means_over_trial_time(subjects, trial_funcs, ylabel, show_legend=True):
  """ Produces a plot of a quantity over trial time,
  averaged over all trials and then over all subjects.
      
  Trial func takes in a (trackit_trial, eyetrack_trial) pair
  and returns the desired quantity for a single trial.
  """
  handles = []
  legend_names = []
  next_color_idx = 0
  if not isinstance(trial_funcs, (list,)): # If the user input only one trial func, reformat as list with empty name
    trial_funcs = [('', trial_funcs)]

  max_time = 600 # Plot up to 10 seconds of trial time

  for (trial_func_name, trial_func) in trial_funcs:
    for experiment_type in conditions:
      subject_values = []
      for subject in subjects.values():
        if not experiment_type in subject.experiments:
          continue
        experiment = subject.experiments[experiment_type]
        if not experiment.has_all_experiment_data:
          print('Excluding subject ' + subject.ID + ' experiment ' + experiment.ID + ' due to missing data type.')
          continue
        trackit, eyetrack = experiment.datatypes['trackit'], experiment.datatypes['eyetrack']
        trial_values = []
        for (trackit_trial, eyetrack_trial) in zip(trackit.trials, eyetrack.trials):
          trial_values.append(trial_func(trackit_trial, eyetrack_trial))
        trial_values = np.asarray([x for x in izip_longest(*trial_values, fillvalue=float('nan'))])
        subject_values.append(np.nanmean(trial_values, axis=1))
      experiment_value = np.asarray([x for x in izip_longest(*subject_values, fillvalue=float('nan'))])
      handles.append(plot_with_std_error_bands(np.linspace(0, (max_time-1)/60, max_time), experiment_value[:max_time,], next_color_idx))
      next_color_idx += 1
      legend_names.append(trial_func_name + ' ' + experiment_type)
  if show_legend:
    plt.legend(handles, legend_names, fontsize=14, loc='lower right')
  plt.xlabel('Trial Time (seconds)', fontsize=14)
  plt.ylabel(ylabel, fontsize=14)
  return handles

def get_experiment_mean_over_time(experiment, trial_func):
  trackit, eyetrack = experiment.datatypes['trackit'], experiment.datatypes['eyetrack']
  trial_values = []
  for (trackit_trial, eyetrack_trial) in zip(trackit.trials, eyetrack.trials):
    trial_values.append(trial_func(trackit_trial, eyetrack_trial))
  trial_values = np.asarray([x for x in izip_longest(*trial_values, fillvalue=float('nan'))])
  return np.nanmean(trial_values, axis=1)

def within_subjects_diff_means_over_trial_time(subjects, trial_funcs, shrinky_minus_noshrinky=True):
  """ Produces a plot of a quantity over trial time,
  averaged over all trials and then over all subjects.
      
  Trial func takes in a (trackit_trial, eyetrack_trial) pair
  and returns the desired quantity for a single trial.
  """
  handles = []
  legend_names = []
  next_color_idx = 0
  max_len = 0

  for (trial_func_name, trial_func) in trial_funcs:
    subject_values = []
    for subject in subjects.values():
      shrinky_value = get_experiment_mean_over_time(subject.experiments['shrinky'], trial_func)
      noshrinky_value = get_experiment_mean_over_time(subject.experiments['noshrinky'], trial_func)
      min_len = min(shrinky_value.shape[0], noshrinky_value.shape[0])
      max_len = max(max_len, min_len)
      if shrinky_minus_noshrinky:
        subject_values.append(shrinky_value[:min_len] - noshrinky_value[:min_len])
        plt.ylabel('Shrinky - No Shrinky')
      else:
        subject_values.append(noshrinky_value[:min_len] - shrinky_value[:min_len])
        plt.ylabel('No Shrinky - Shrinky')
    experiment_value = np.asarray([x for x in izip_longest(*subject_values, fillvalue=float('nan'))])
    handles.append(plot_with_std_error_bands(range(experiment_value.shape[0]), experiment_value, next_color_idx))
    next_color_idx += 1
    legend_names.append(trial_func_name)
  plt.legend(handles, legend_names)
  plt.xlabel('Trial Time (seconds)', fontsize=20)
  # plt.plot([0, max_len], [0, 0], color='black', linestyle='--')
  return handles

def corrcoef_with_bootstrapped_CIs(xs, ys):
  num_bootstrap_repetitions = 2000
  assert ys.shape[0] == xs.shape[0], \
    'Equal sample sizes are required to compute correlations. ' + \
    'However, the sample sizes are ' + str(ys.shape[0]) + ' and ' + str(xs.shape[0]) + '.'
  sample_size = xs.shape[0]
  corrs_bootstrapped = np.zeros((num_bootstrap_repetitions))
  for bootstrap_rep in range(num_bootstrap_repetitions):
    sample_idxs = np.random.choice(sample_size, size=sample_size, replace=True)
    corrs_bootstrapped[bootstrap_rep] = np.corrcoef(xs[sample_idxs], ys[sample_idxs])[0,1]
  corr_lower = np.nanpercentile(corrs_bootstrapped, 2.5)
  corr_upper = np.nanpercentile(corrs_bootstrapped, 97.5)
  return np.corrcoef(xs, ys)[0,1], corr_lower, corr_upper

def reduce_to_corr_over_experiments(subjects, x_func, y_funcs, xlabel, ylabel):
  """ Produces a scatter plot of particular x-values against (a selection of) y-values.
  Each point corresponds to a single experiment.
      
  Trial func takes in an experiment and returns the desired quantity for the entire experiment.
  """
  handles = []
  legend_names = []
  next_color_idx = 0
  if not isinstance(y_funcs, (list,)): # If the user input only one trial func, reformat as list with empty name
    y_funcs = [('', y_funcs)]

  for (y_func_name, y_func) in y_funcs:
    for experiment_type in conditions:
      subject_values = []
      for subject in subjects.values():
        if not experiment_type in subject.experiments:
          continue
        experiment = subject.experiments[experiment_type]
        if not experiment.has_all_experiment_data:
          print('Excluding subject ' + subject.ID + ' experiment ' + experiment.ID + ' due to missing data type.')
          continue
        subject_values.append((x_func(experiment), y_func(experiment)))
      xs, ys = zip(*subject_values)
      handles.append(plt.scatter(xs, ys))
      next_color_idx += 1
      legend_names.append(y_func_name + ' ' + experiment_type)
      # corrcoef, corrcoef_lower, corrcoef_upper = 
      print('The correlation with behavioral for ' + y_func_name + ' ' + experiment_type + ' is ' + \
            '%.2f, with 95%% CI (%.2f, %.2f).' % corrcoef_with_bootstrapped_CIs(np.array(xs), np.array(ys)))
  plt.legend(handles, legend_names)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  return handles

def reduce_to_means_over_trial_num(subjects, trial_funcs, ylabel, start_color_idx=0):
  """ Produces a plot of a quantity over trial number,
  averaged over all (non-NaN) timepoints and then over all subjects.
      
  Trial func takes in a (trackit_trial, eyetrack_trial) pair
  and returns the desired quantity for a single trial.
  """
  handles = []
  legend_names = []
  next_color_idx = start_color_idx
  if not isinstance(trial_funcs, (list,)): # If the user input only one trial func, reformat as list with empty name
    trial_funcs = [('', trial_funcs)]

  for (trial_func_name, trial_func) in trial_funcs:
    for experiment_type in conditions:
      subject_values = []
      for subject in subjects.values():
        if not experiment_type in subject.experiments:
          continue
        experiment = subject.experiments[experiment_type]
        if not experiment.has_all_experiment_data:
          print('Excluding subject ' + subject.ID + ' experiment ' + experiment.ID + ' due to missing data type.')
          continue
        trackit, eyetrack = experiment.datatypes['trackit'], experiment.datatypes['eyetrack']
        trial_values = []
        for (trackit_trial, eyetrack_trial) in zip(trackit.trials, eyetrack.trials):
          trial_values.append(trial_func(trackit_trial, eyetrack_trial))
        trial_values = np.asarray([x for x in izip_longest(*trial_values, fillvalue=float('nan'))])
        subject_values.append(np.nanmean(trial_values, axis=0))
      experiment_value = np.asarray([x for x in izip_longest(*subject_values, fillvalue=float('nan'))])
      handles.append(plot_with_std_error_bars(range(experiment_value.shape[0])[1:], experiment_value[1:,], next_color_idx))
      next_color_idx += 1
      legend_names.append(trial_func_name + ' ' + experiment_type)
  plt.legend(handles, legend_names)
  plt.xlabel('Trial Number')
  plt.ylabel(ylabel)
  return handles

def distance_from_target(trackit_trial, eyetrack_trial):
  target_positions = trackit_trial.object_positions[trackit_trial.target_index, :, :]
  return np.sqrt(np.square(target_positions - eyetrack_trial.data[:,1:]).sum(axis=1))

def performance_according_to_HMM(trackit_trial, eyetrack_trial, treat_missing_data_as_incorrect = False):
  if not hasattr(eyetrack_trial, 'HMM_MLE'): # If HMM MLE is not already cached, compute it
    rint('Computing new trial HMM...')
    eyetrack_positions = eyetrack_trial.data[:, 1:]
    sigma2_child = 250 ** 2 # Value taken from supervised results in CogSci 18 paper
    # sigma2_child = 870 ** 2 # Value taken from supervised results in CogSci 18 paper
    eyetrack_trial.HMM_MLE = eyetracking_hmm.get_MLE(eyetrack_positions, trackit_trial.object_positions, sigma2 = sigma2_child)
  if treat_missing_data_as_incorrect:
    return eyetrack_trial.HMM_MLE == trackit_trial.target_index
  else:
    def on_target_or_nan(x): # replace missing data (encoded as -1) with nan
      if x == -1:
        return float("nan")
      return x == trackit_trial.target_index
    return [on_target_or_nan(x) for x in eyetrack_trial.HMM_MLE]

def behavioral_performance(trackit_trial, eyetrack_trial):
  return np.array('true' == trackit_trial.trial_metadata['gridClickCorrect']).reshape(1,)

def missing_data(trackit_trial, eyetrack_trial):
  eyetrack_positions = eyetrack_trial.data[:, 1]
  return np.isnan(eyetrack_positions)

def performance_according_to_naive(trackit_trial, eyetrack_trial, treat_missing_data_as_incorrect = False):
  if not hasattr(eyetrack_trial, 'naive_MLE'): # If naive MLE is not already cached, compute it
    # print('Computing new trial naive...')
    eyetrack = eyetrack_trial.data[:, 1:]

    # naive_eyetracking.get_trackit_MLE() expects the target and distractor positions as separate arguments
    target = trackit_trial.object_positions[trackit_trial.target_index,:,:]
    distractors = np.delete(trackit_trial.object_positions, trackit_trial.target_index, axis=0)

    eyetrack_trial.naive_MLE = naive_eyetracking.get_trackit_MLE(eyetrack.swapaxes(0, 1), target.swapaxes(0, 1), distractors.swapaxes(1, 2))

  if treat_missing_data_as_incorrect:
    return eyetrack_trial.naive_MLE == trackit_trial.target_index
  else:
    def on_target_or_nan(x): # replace missing data (encoded as -1) with nan
      if x == -1:
        return float("nan")
      return x == trackit_trial.target_index
    return [on_target_or_nan(x) for x in eyetrack_trial.naive_MLE]

def missing_data(trackit_trial, eyetrack_trial):
  eyetrack_positions = eyetrack_trial.data[:, 1]
  return np.isnan(eyetrack_positions)

# Interpolate missing eyetracking data and store new imputed data proportion
def impute_missing_data(experiment, max_len = 10):
  if not 'eyetrack' in experiment.datatypes:
    return # If eyetracking data is missing, nothing to do
  eyetrack = experiment.datatypes['eyetrack']
  util.impute_missing_data_D(eyetrack.raw_data.T, max_len=max_len).T
  eyetrack.proportion_missing_frames_after_imputation = np.mean(np.isnan(eyetrack.raw_data[:, 1]))
  eyetrack.proportion_imputed_frames = eyetrack.proportion_missing_frames_after_imputation - eyetrack.proportion_total_missing_frames

# Break experiment's eyetracking data into trialsprint('After: ' + str(trackit_trial.object_positions.shape))
def break_eyetracking_into_trials(experiment):
  if not 'trackit' in experiment.datatypes or not 'eyetrack' in experiment.datatypes:
    experiment.has_all_experiment_data = False
    return # If either TrackIt or eyetracking data is missing, nothing to do
  experiment.has_all_experiment_data = True
  trackit, eyetrack = experiment.datatypes['trackit'], experiment.datatypes['eyetrack']
  eyetrack.trials = []
  for (trial_idx, trial) in enumerate(trackit.trials):
    trial_start, trial_end = trial.timestamps[0], trial.timestamps[-1]
    trial_eyetrack_data = np.asarray([frame for frame in eyetrack.raw_data if trial_start < frame[0] and frame[0] < trial_end])
    eyetrack.trials.append(ls.Eyetrack_Trial_Data(trial_eyetrack_data))
    eyetrack.trials[-1].proportion_missing_frames = np.mean(np.isnan(trial_eyetrack_data[:,1]))

# Interpolate the TrackIt data points to be synchronized with the Eyetracking data
def interpolate_trackit_to_eyetracking(experiment):
  if not experiment.has_all_experiment_data:
    return  # If either TrackIt or eyetracking data is missing, nothing to do
  trackit, eyetrack = experiment.datatypes['trackit'], experiment.datatypes['eyetrack']
  for (trackit_trial, eyetrack_trial) in zip(trackit.trials, eyetrack.trials):
    eyetrack_len = eyetrack_trial.data.shape[0]
    interpolated_object_positions = np.zeros((trackit_trial.object_positions.shape[0], eyetrack_len, 2))
    # X coordinates
    interpolated_object_positions[:, :, 0] = util.interpolate_to_length_D(trackit_trial.object_positions[:, :, 0], new_len=eyetrack_len)
    # Y coordinates
    interpolated_object_positions[:, :, 1] = util.interpolate_to_length_D(trackit_trial.object_positions[:, :, 1], new_len=eyetrack_len)
    trackit_trial.object_positions = interpolated_object_positions

# Annotates experiment with the trials to be filtered, as well as whether the entire experiment should be filtered.
# Also excludes practice trials.
def filter_experiment(experiment, min_prop_data_per_trial=0.5, min_prop_trials_per_subject=0.5):
  try:
    eyetrack = experiment.datatypes['eyetrack']
  except KeyError: # If the experiment doesn't have eyetracking data, nothing to do
    return
  except AttributeError as e:
    print("AttributeError: " + str(e))
    print('Perhaps, the eyetracking data has not yet been broken into trials. Run break_eyetracking_into_trials(experiment) first.')
    return
  trials = eyetrack.trials
  experiment.trials_to_keep = [idx for (idx, trial) in enumerate(trials) \
                                  if 1 - trial.proportion_missing >= min_prop_data_per_trial and idx > 0]
  experiment.keep_experiment = (len(experiment.trials_to_keep) >= len(trials) * min_prop_trials_per_subject)

# Given a list X, returns a list of changepoints
def get_changepoints(X):
  return X[:-1] != X[1:]
