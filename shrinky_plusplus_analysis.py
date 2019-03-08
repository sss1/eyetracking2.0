import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle
import math

np.set_printoptions(threshold=np.nan)

# Imports used for linear regression
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from scipy import stats


import load_subjects as ls
import eyetracking2_util as util

cache_file = 'cache/sigma250.cache'

try: # If the data preprocessed data have already been cached, just load them.
  with open(cache_file , 'r') as f:
    subjects = pickle.load(f)
  data_cached = True

except IOError:
  # No data cache found. Load all data and run analyses from scratch.
  data_cached = False

  # Load all experiment data
  subjects = ls.load_dataset('shrinky', 'eyetrack')
  subjects = ls.load_dataset('shrinky', 'trackit', subjects)
  subjects = ls.load_dataset('noshrinky', 'eyetrack', subjects)
  subjects = ls.load_dataset('noshrinky', 'trackit', subjects)

  # Combine eyetracking with trackit data and perform all preprocessing
  for subject in subjects.values():
    for (experiment_ID, experiment) in subject.experiments.items():
      # TODO: Have each preprocessing step add an annotation to experiment object
      ls.add_age(experiment)
      if experiment.age > 6.0:
        print('Warning: Subject ' + subject.ID + ' experiment ' + experiment_ID + ' has age ' + str(experiment.age))
      util.impute_missing_data(experiment)
      util.break_eyetracking_into_trials(experiment)
      util.interpolate_trackit_to_eyetracking(experiment)
      util.filter_experiment(experiment)
      # try:
      #   print('Subject ' + subject.ID + ' provided ' + str(len(experiment.trials_to_keep)) + ' good trials in condition ' + experiment_ID)
      # except AttributeError:
      #   pass

def subject_is_good(subject):
  return 'shrinky' in subject.experiments and \
         'noshrinky' in subject.experiments and \
         'trackit' in subject.experiments['shrinky'].datatypes and \
         'eyetrack' in subject.experiments['shrinky'].datatypes and \
         'trackit' in subject.experiments['noshrinky'].datatypes and \
         'eyetrack' in subject.experiments['noshrinky'].datatypes and \
         subject.experiments['shrinky'].datatypes['trackit'].metadata['Grid X Size'] == '6' and \
         subject.experiments['shrinky'].datatypes['trackit'].metadata['Grid Y Size'] == '6' and \
         subject.experiments['noshrinky'].datatypes['trackit'].metadata['Grid X Size'] == '6' and \
         subject.experiments['noshrinky'].datatypes['trackit'].metadata['Grid Y Size'] == '6' and \
         len(subject.experiments['shrinky'].trials_to_keep) >= 5 and \
         len(subject.experiments['noshrinky'].trials_to_keep) >= 5# and \
                # (subject.experiments['shrinky'].age + subject.experiments['shrinky'].age) / 2 > 4.340862422997897

# Filter out subjects based on missing data, incorrect conditions, etc.
good_subjects = { subject_ID : subject for (subject_ID, subject) in subjects.items() if subject_is_good(subject) }
num_good_subjects = len(good_subjects)
print(str(num_good_subjects) + ' good subjects: ' + str(good_subjects.keys()))
bad_subjects = set(subjects.keys()) - set(good_subjects.keys())
print(str(len(bad_subjects)) + ' bad subjects: ' + str(bad_subjects))
# TODO: Print out basic stats of why subjects were discarded

trials = range(1, 11) # Ignore practice trial (Trial 0)

# Histogram of participant ages, before and after filtering
plt.figure(9)
plt.xlim((3,6))
plt.ylim((0,20))
plt.title('Distribution of Participant Ages Before and After Filtering')
all_ages = np.array([experiment.age for subject in subjects.values() for experiment in subject.experiments.values()])
plt.hist(all_ages, bins = np.linspace(3.0, 6.0, 30), label='Before Filtering')
print('Ages of all participants:', 'median:', np.median(all_ages), 'mean:', np.mean(all_ages), 'sd:', np.std(all_ages), '.')
good_ages = np.array([experiment.age for subject in good_subjects.values() for experiment in subject.experiments.values()])
plt.hist(good_ages, bins = np.linspace(3.0, 6.0, 30), label='After Filtering')
print('Ages of good participants:', 'median:', np.median(good_ages), 'mean:', np.mean(good_ages), 'sd:', np.std(good_ages), '.')
good_ages_median = np.median(good_ages)
plt.legend()
plt.xlabel('Age')
plt.ylabel('Number of Participants')

# Calculate mean performance according to HMM, mean performance according to Naive, and mean (within-subjects) difference in performances
shrinky_HMM_performance_by_subject = np.array([np.nanmean([np.nanmean(util.performance_according_to_HMM(*trial))
                                        for trial in zip(subject.experiments['shrinky'].datatypes['trackit'].trials,
                                                         subject.experiments['shrinky'].datatypes['eyetrack'].trials)])
                                            for subject in good_subjects.values()])
noshrinky_HMM_performance_by_subject = np.array([np.nanmean([np.nanmean(util.performance_according_to_HMM(*trial))
                                        for trial in zip(subject.experiments['noshrinky'].datatypes['trackit'].trials,
                                                         subject.experiments['noshrinky'].datatypes['eyetrack'].trials)])
                                            for subject in good_subjects.values()])
diffs_HMM_performance_by_subject = shrinky_HMM_performance_by_subject - noshrinky_HMM_performance_by_subject
print('Mean performance according to HMM:', np.nanmean(shrinky_HMM_performance_by_subject), 1.96 * np.std(shrinky_HMM_performance_by_subject) / math.sqrt(num_good_subjects))
# print('Mean performance according to Naive:', np.nanmean(noshrinky_HMM_performance_by_subject), 1.96 * np.std(noshrinky_HMM_performance_by_subject) / math.sqrt(num_good_subjects))
# print('Mean difference in performances (HMM - Naive):', np.nanmean(diffs_HMM_performance_by_subject), 1.96 * np.std(diffs_HMM_performance_by_subject) / math.sqrt(num_good_subjects))

# Code for printing HMM-Naive difference
# x = 0
# for subject in good_subjects.values():
#   for experiment in subject.experiments.values():
#     cond_diff = []
#     cond_missing = []
#     for (trial_num, (trackit_trial, eyetrack_trial)) in enumerate(zip(experiment.datatypes['trackit'].trials, experiment.datatypes['eyetrack'].trials)):
#       cond_diff.append(np.nanmean(util.performance_according_to_HMM(trackit_trial, eyetrack_trial)) \
#                 - np.nanmean(util.performance_according_to_naive(trackit_trial, eyetrack_trial)))
#       cond_missing.append(eyetrack_trial.proportion_missing)
#     mean_diff = np.mean(cond_diff)
#     mean_missing = np.mean(cond_missing)
#     # if mean_diff > 0.235 and mean_missing < 0.1:
#     # print('Subject:', subject.ID, 'Experiment:', experiment.ID)
#     # print('Subject:', subject.ID, 'Experiment:', experiment.ID, 'Missing:', mean_missing)
#     # if mean_missing < 0.05:
#     # if mean_diff > 0.25:# and mean_missing < 0.1:
#     print('Subject:', subject.ID, 'Experiment:', experiment.ID, 'Diff:', mean_diff, 'Missing:', mean_missing)
#     x += 1
# print('Number of subjects printed:', x)

# # Plots of eyetracking distance from target over trial time and trial number
# plt.figure(0)
# plt.title('Distance from Target over Trial Time, Shrinky and Noshrinky \n(Missing Data removed)')
# util.reduce_to_means_over_trial_time(good_subjects, util.distance_from_target, ylabel='Mean Distance from Target')
# plt.plot([600, 600], [0, 1000], linestyle = '--', color = 'k')
# plt.figure(1)
# plt.title('Distance from Target over Trial Number, Shrinky and Noshrinky \n(Missing Data removed)')
# util.reduce_to_means_over_trial_num(good_subjects, util.distance_from_target, ylabel='Mean Distance from Target')
# plt.ylim(bottom=0)

# # Plots of eyetracking performance over trial time and trial number, counting missing data as incorrect ("worst case")
# performance_according_to_HMM_worst_case = lambda x, y : util.performance_according_to_HMM(x, y, treat_missing_data_as_incorrect = True)
# performance_according_to_naive_worst_case = lambda x, y : util.performance_according_to_naive(x, y, treat_missing_data_as_incorrect = True)
# plt.figure(2)
# plt.title('HMM and Naive Performance over Trial Time, Shrinky and Noshrinky\n(Missing Data treated as incorrect)')
# util.reduce_to_means_over_trial_time(good_subjects,
#                                      [('HMM', performance_according_to_HMM_worst_case), ('Naive', performance_according_to_naive_worst_case)],
#                                      ylabel='Mean Proportion of Trials Present and on Target')
# plt.plot([600, 600], [0, 1], linestyle = '--', color = 'k')
# plt.ylim([0, 1])
# plt.figure(3)
# plt.title('HMM and Naive Performance over Trial Number, Shrinky and Noshrinky\n(Missing Data treated as incorrect)')
# util.reduce_to_means_over_trial_num(good_subjects,
#                                     [('HMM', util.performance_according_to_HMM), ('Naive', util.performance_according_to_naive)],
#                                     ylabel='Mean Proportion of Frames Present and on Target')
# plt.ylim([0, 1])

# Plots of eyetracking performance over trial time and trial number, over non-missing data ("average case")
plt.figure(20, figsize=(3, 3))
# plt.title('HMM Performance over Trial Time')
# Median-split subjects into younger and older groups
younger_subjects = {subject_ID : subject for (subject_ID, subject) in good_subjects.items()
                        if (subject.experiments['shrinky'].age + subject.experiments['shrinky'].age) / 2 < good_ages_median}
older_subjects = {subject_ID : subject for (subject_ID, subject) in good_subjects.items()
                        if (subject.experiments['shrinky'].age + subject.experiments['shrinky'].age) / 2 > good_ages_median}
plt.title('Younger Children', fontsize=14)
plt.ylim([0, 1])
util.reduce_to_means_over_trial_time(younger_subjects,
                                     [('HMM', util.performance_according_to_HMM)],#, ('Naive', util.performance_according_to_naive)],
                                     ylabel='HMM Performance',
                                     show_legend=False)
plt.tight_layout()
plt.figure(21, figsize=(3, 3))
plt.title('Older Children', fontsize=14)
plt.ylim([0, 1])
util.reduce_to_means_over_trial_time(older_subjects,
                                     [('HMM', util.performance_according_to_HMM)],#, ('Naive', util.performance_according_to_naive)],
                                     ylabel='',
                                     show_legend=False)
plt.legend(['Exogenous', 'Endogenous'], fontsize=14)
plt.tight_layout()
# plt.plot([600, 600], [0, 1], linestyle = '--', color = 'k')
plt.figure(22, figsize=(6, 3))
plt.xlabel('Age', fontsize=15)
plt.ylabel(r'$\beta_{time}$', fontsize=18)
ages = np.zeros((num_good_subjects,))
shrinky_betas_0 = np.zeros((num_good_subjects,))
shrinky_betas_time = np.zeros((num_good_subjects,))
noshrinky_betas_0 = np.zeros((num_good_subjects,))
noshrinky_betas_time = np.zeros((num_good_subjects,))
max_trial_len = 601
time_xs = np.linspace(0, (max_trial_len - 1)/60.0, max_trial_len)
for (subject_idx, subject) in enumerate(good_subjects.values()):
  ys = np.nanmean(np.array([util.performance_according_to_HMM(subject.experiments['shrinky'].datatypes['trackit'].trials[trial_idx],
                                                              subject.experiments['shrinky'].datatypes['eyetrack'].trials[trial_idx]
                            )[:max_trial_len] for trial_idx in trials]), axis=0)
  shrinky_betas_0[subject_idx], shrinky_betas_time[subject_idx] = sm.OLS(ys, sm.add_constant(time_xs)).fit().params
  ys = np.nanmean(np.array([util.performance_according_to_HMM(subject.experiments['noshrinky'].datatypes['trackit'].trials[trial_idx],
                                                              subject.experiments['noshrinky'].datatypes['eyetrack'].trials[trial_idx]
                            )[:max_trial_len] for trial_idx in trials]), axis=0)
  noshrinky_betas_0[subject_idx], noshrinky_betas_time[subject_idx] = sm.OLS(ys, sm.add_constant(time_xs)).fit().params
  ages[subject_idx] = subject.experiments['shrinky'].age
plt.scatter(ages, shrinky_betas_time, c='C0')
shrinky_betas_time_over_age = sm.OLS(shrinky_betas_time, sm.add_constant(ages)).fit()
age_range = np.array([3.5, 6])
plt.plot(age_range, shrinky_betas_time_over_age.predict(sm.add_constant(age_range)), c='C0')
print(shrinky_betas_time_over_age.summary())
plt.scatter(ages, noshrinky_betas_time, c='C1')
noshrinky_betas_time_over_age = sm.OLS(noshrinky_betas_time, sm.add_constant(ages)).fit()
plt.plot(age_range, noshrinky_betas_time_over_age.predict(sm.add_constant(age_range)), c='C1')
print(noshrinky_betas_time_over_age.summary())
plt.tight_layout()

# plt.figure(5)
# Since performance measures across trials are correlated within subjects,
# rather than using the usual significance test for linear regression,
# we perform regression for each subject, and then use a t-test for whether
# the mean betas are zero.
HMM_shrinky_betas_0 = np.zeros((num_good_subjects,))
HMM_shrinky_betas_trial = np.zeros((num_good_subjects,))
HMM_noshrinky_betas_0 = np.zeros((num_good_subjects,))
HMM_noshrinky_betas_trial = np.zeros((num_good_subjects,))
behavioral_shrinky_betas_0 = np.zeros((num_good_subjects,))
behavioral_shrinky_betas_trial = np.zeros((num_good_subjects,))
behavioral_noshrinky_betas_0 = np.zeros((num_good_subjects,))
behavioral_noshrinky_betas_trial = np.zeros((num_good_subjects,))
for (subject_idx, subject) in enumerate(good_subjects.values()):
  HMM_shrinky_betas_0[subject_idx], HMM_shrinky_betas_trial[subject_idx] = \
    sm.OLS([np.nanmean(util.performance_according_to_HMM(subject.experiments['shrinky'].datatypes['trackit'].trials[trial_idx],
                                                         subject.experiments['shrinky'].datatypes['eyetrack'].trials[trial_idx]))
        for trial_idx in trials], sm.add_constant(trials)).fit().params
  HMM_noshrinky_betas_0[subject_idx], HMM_noshrinky_betas_trial[subject_idx] = \
    sm.OLS([np.nanmean(util.performance_according_to_HMM(subject.experiments['noshrinky'].datatypes['trackit'].trials[trial_idx],
                                                         subject.experiments['noshrinky'].datatypes['eyetrack'].trials[trial_idx]))
        for trial_idx in trials], sm.add_constant(trials)).fit().params
  behavioral_shrinky_betas_0[subject_idx], behavioral_shrinky_betas_trial[subject_idx] = \
    sm.OLS([util.behavioral_performance(subject.experiments['shrinky'].datatypes['trackit'].trials[trial_idx],
                                        subject.experiments['shrinky'].datatypes['eyetrack'].trials[trial_idx])
        for trial_idx in trials], sm.add_constant(trials)).fit().params
  behavioral_noshrinky_betas_0[subject_idx], behavioral_noshrinky_betas_trial[subject_idx] = \
    sm.OLS([util.behavioral_performance(subject.experiments['noshrinky'].datatypes['trackit'].trials[trial_idx],
                                        subject.experiments['noshrinky'].datatypes['eyetrack'].trials[trial_idx])
        for trial_idx in trials], sm.add_constant(trials)).fit().params

HMM_shrinky_beta_0_mean = np.nanmean(HMM_shrinky_betas_0)
HMM_shrinky_beta_0_CI = 1.96 * np.nanstd(HMM_shrinky_betas_0)/math.sqrt(num_good_subjects)
print('\nHMM_shrinky_beta_0: mean: ' + str(HMM_shrinky_beta_0_mean) + \
      ' 95% CI: (' + str(HMM_shrinky_beta_0_mean - HMM_shrinky_beta_0_CI) + ', ' + str(HMM_shrinky_beta_0_mean + HMM_shrinky_beta_0_CI) + ').')
print(stats.ttest_1samp(HMM_shrinky_betas_0[~np.isnan(HMM_shrinky_betas_0)], 0))
HMM_shrinky_beta_trial_mean = np.nanmean(HMM_shrinky_betas_trial)
HMM_shrinky_beta_trial_CI = 1.96 * np.nanstd(HMM_shrinky_betas_trial)/math.sqrt(num_good_subjects)
print('HMM_shrinky_beta_trial: mean: ' + str(HMM_shrinky_beta_trial_mean) + \
      '95% CI: (' + str(HMM_shrinky_beta_trial_mean - HMM_shrinky_beta_trial_CI) + ', ' + \
                    str(HMM_shrinky_beta_trial_mean + HMM_shrinky_beta_trial_CI) + ').')
print(stats.ttest_1samp(HMM_shrinky_betas_trial[~np.isnan(HMM_shrinky_betas_trial)], 0))
HMM_noshrinky_beta_0_mean = np.nanmean(HMM_noshrinky_betas_0)
HMM_noshrinky_beta_0_CI = 1.96 * np.nanstd(HMM_noshrinky_betas_0)/math.sqrt(num_good_subjects)
print('\nHMM_noshrinky_beta_0: mean: ' + str(HMM_noshrinky_beta_0_mean) + \
      ' 95% CI: (' + str(HMM_noshrinky_beta_0_mean - HMM_noshrinky_beta_0_CI) + ', ' + str(HMM_noshrinky_beta_0_mean + HMM_noshrinky_beta_0_CI) + ').')
print(stats.ttest_1samp(HMM_noshrinky_betas_0[~np.isnan(HMM_noshrinky_betas_0)], 0))
HMM_noshrinky_beta_trial_mean = np.nanmean(HMM_noshrinky_betas_trial)
HMM_noshrinky_beta_trial_CI = 1.96 * np.nanstd(HMM_noshrinky_betas_trial)/math.sqrt(num_good_subjects)
print('HMM_noshrinky_beta_trial: mean: ' + str(HMM_noshrinky_beta_trial_mean) + \
      '95% CI: (' + str(HMM_noshrinky_beta_trial_mean - HMM_noshrinky_beta_trial_CI) + ', ' + \
                    str(HMM_noshrinky_beta_trial_mean + HMM_noshrinky_beta_trial_CI) + ').')
print(stats.ttest_1samp(HMM_noshrinky_betas_trial[~np.isnan(HMM_noshrinky_betas_trial)], 0))
behavioral_shrinky_beta_0_mean = np.nanmean(behavioral_shrinky_betas_0)
behavioral_shrinky_beta_0_CI = 1.96 * np.nanstd(behavioral_shrinky_betas_0)/math.sqrt(num_good_subjects)
print('\nbehavioral_shrinky_beta_0: mean: ' + str(behavioral_shrinky_beta_0_mean) + \
      ' 95% CI: (' + str(behavioral_shrinky_beta_0_mean - behavioral_shrinky_beta_0_CI) + ', ' + str(behavioral_shrinky_beta_0_mean + behavioral_shrinky_beta_0_CI) + ').')
print(stats.ttest_1samp(behavioral_shrinky_betas_0[~np.isnan(behavioral_shrinky_betas_0)], 0))
behavioral_shrinky_beta_trial_mean = np.nanmean(behavioral_shrinky_betas_trial)
behavioral_shrinky_beta_trial_CI = 1.96 * np.nanstd(behavioral_shrinky_betas_trial)/math.sqrt(num_good_subjects)
print('behavioral_shrinky_beta_trial: mean: ' + str(behavioral_shrinky_beta_trial_mean) + \
      '95% CI: (' + str(behavioral_shrinky_beta_trial_mean - behavioral_shrinky_beta_trial_CI) + ', ' + \
                    str(behavioral_shrinky_beta_trial_mean + behavioral_shrinky_beta_trial_CI) + ').')
print(stats.ttest_1samp(behavioral_shrinky_betas_trial[~np.isnan(behavioral_shrinky_betas_trial)], 0))
behavioral_noshrinky_beta_0_mean = np.nanmean(behavioral_noshrinky_betas_0)
behavioral_noshrinky_beta_0_CI = 1.96 * np.nanstd(behavioral_noshrinky_betas_0)/math.sqrt(num_good_subjects)
print('\nbehavioral_noshrinky_beta_0: mean: ' + str(behavioral_noshrinky_beta_0_mean) + \
      ' 95% CI: (' + str(behavioral_noshrinky_beta_0_mean - behavioral_noshrinky_beta_0_CI) + ', ' + str(behavioral_noshrinky_beta_0_mean + behavioral_noshrinky_beta_0_CI) + ').')
print(stats.ttest_1samp(behavioral_noshrinky_betas_0[~np.isnan(behavioral_noshrinky_betas_0)], 0))
behavioral_noshrinky_beta_trial_mean = np.nanmean(behavioral_noshrinky_betas_trial)
behavioral_noshrinky_beta_trial_CI = 1.96 * np.nanstd(behavioral_noshrinky_betas_trial)/math.sqrt(num_good_subjects)
print('behavioral_noshrinky_beta_trial: mean: ' + str(behavioral_noshrinky_beta_trial_mean) + \
      '95% CI: (' + str(behavioral_noshrinky_beta_trial_mean - behavioral_noshrinky_beta_trial_CI) + ', ' + \
                    str(behavioral_noshrinky_beta_trial_mean + behavioral_noshrinky_beta_trial_CI) + ').')
print(stats.ttest_1samp(behavioral_noshrinky_betas_trial[~np.isnan(behavioral_noshrinky_betas_trial)], 0))


# younger_all_trial_means = np.array([[np.nanmean(util.performance_according_to_HMM(
#                                             subject.experiments['shrinky'].datatypes['trackit'].trials[trial_idx],
#                                             subject.experimentsi['shrinky'].datatypes['eyetrack'].trials[trial_idx]))
#                                         for trial_idx in trials] for subject in younger_subjects.values()])
# younger_trial_mean_means = np.mean(younger_all_trial_means, axis=0)
# younger_trial_mean_CIs = 1.95 * np.std(younger_all_trial_means, axis=0)
plt.figure(23)
util.reduce_to_means_over_trial_num(younger_subjects, [('HMM', util.performance_according_to_HMM)], ylabel='Mean Performance')
plt.ylim((0,1))
plt.figure(24)
util.reduce_to_means_over_trial_num(older_subjects, [('HMM', util.performance_according_to_HMM)], ylabel='Mean Performance')
plt.ylim((0,1))

# trial_nums = []
# HMM_shrinky_trial_performances_by_subject = []
# HMM_noshrinky_trial_performances_by_subject = []
# behavioral_shrinky_trial_performances_by_subject = []
# behavioral_noshrinky_trial_performances_by_subject = []
# for trial_idx in range(1, 11):
#   for subject in good_subjects.values():
#     trial_nums.append(trial_idx)
#     trackit_trial = subject.experiments['shrinky'].datatypes['trackit'].trials[trial_idx]
#     eyetrack_trial = subject.experiments['shrinky'].datatypes['eyetrack'].trials[trial_idx]
#     HMM_shrinky_trial_performances_by_subject.append(np.nanmean(util.performance_according_to_HMM(trackit_trial, eyetrack_trial)))
#     behavioral_shrinky_trial_performances_by_subject.append(util.behavioral_performance(trackit_trial, eyetrack_trial))
# 
#     trackit_trial = subject.experiments['noshrinky'].datatypes['trackit'].trials[trial_idx]
#     eyetrack_trial = subject.experiments['noshrinky'].datatypes['eyetrack'].trials[trial_idx]
#     HMM_noshrinky_trial_performances_by_subject.append(np.nanmean(util.performance_according_to_HMM(trackit_trial, eyetrack_trial)))
#     behavioral_noshrinky_trial_performances_by_subject.append(util.behavioral_performance(trackit_trial, eyetrack_trial))
# trial_nums = np.array(trial_nums)
# HMM_shrinky_trial_performances_by_subject = np.array(HMM_shrinky_trial_performances_by_subject)
# HMM_noshrinky_trial_performances_by_subject = np.array(HMM_noshrinky_trial_performances_by_subject)
# behavioral_shrinky_trial_performances_by_subject = np.array(behavioral_shrinky_trial_performances_by_subject)
# behavioral_noshrinky_trial_performances_by_subject = np.array(behavioral_noshrinky_trial_performances_by_subject)
#   
# plt.subplot(1, 2, 1)
# plt.ylim((0,1))
# 
# 
# plt.title('HMM Performance over Trial Number')
# # plt.scatter(trial_nums, HMM_shrinky_trial_performances_by_subject)
# reg = sm.OLS(HMM_shrinky_trial_performances_by_subject[~np.isnan(HMM_shrinky_trial_performances_by_subject)],
#     sm.add_constant(trial_nums[~np.isnan(HMM_shrinky_trial_performances_by_subject)])).fit()
# print('\nRegression of Shrinky HMM Performance over Trial Number:')
# print(reg.summary())
# pred_input = np.linspace(1, 10, 10)
# plt.plot(pred_input,reg.predict(sm.add_constant(pred_input)), c='C0')
# reg = sm.OLS(HMM_noshrinky_trial_performances_by_subject[~np.isnan(HMM_noshrinky_trial_performances_by_subject)],
#     sm.add_constant(trial_nums[~np.isnan(HMM_noshrinky_trial_performances_by_subject)])).fit()
# print('\nRegression of NoShrinky HMM Performance over Trial Number:')
# print(reg.summary())
# print(reg.params[1])
# plt.plot(pred_input,reg.predict(sm.add_constant(pred_input)), c='C1')
# 
# # util.reduce_to_means_over_trial_num(good_subjects,
# #                                     [('HMM', util.performance_according_to_HMM)],# ('Naive', util.performance_according_to_naive)],
# #                                     ylabel='Mean Performance')
# 
# plt.subplot(1, 2, 2)
# plt.ylim((0,1))
# plt.title('Behavioral Performance over Trial Number')
# # plt.scatter(trial_nums, behavioral_shrinky_trial_performances_by_subject)
# reg = sm.OLS(behavioral_shrinky_trial_performances_by_subject, sm.add_constant(trial_nums)).fit()
# print('\nRegression of Shrinky Behavioral Performance over Trial Number:')
# print(reg.summary())
# plt.plot(pred_input,reg.predict(sm.add_constant(pred_input)), c='C4')
# reg = sm.OLS(behavioral_noshrinky_trial_performances_by_subject, sm.add_constant(trial_nums)).fit()
# print('\nRegression of NoShrinky Behavioral Performance over Trial Number:')
# print(reg.summary())
# plt.plot(pred_input,reg.predict(sm.add_constant(pred_input)), c='C5')
# # util.reduce_to_means_over_trial_num(good_subjects,
# #                                     [('HMM', util.performance_according_to_HMM)],# ('Naive', util.performance_according_to_naive)],
# #                                     ylabel='Mean Performance')
# 
# plt.scatter(trial_nums, HMM_noshrinky_trial_performances_by_subject)
# plt.ylim([0, 1])
# plt.subplot(1, 2, 2)
# plt.title('Behavioral Performance over Trial Number,')
# # plt.scatter(trial_nums, behavioral_shrinky_trial_performances_by_subject)
# # plt.scatter(trial_nums, behavioral_noshrinky_trial_performances_by_subject)
# # util.reduce_to_means_over_trial_num(good_subjects,
# #                                     [('Behavioral', util.behavioral_performance)],
# #                                     ylabel='', start_color_idx = 4)
# plt.ylim([0, 1])

# Plots of proportion of missing data over trial time and trial number
plt.figure(6)
plt.title('Proportion of Missing Data over Trial Time')
util.reduce_to_means_over_trial_time(good_subjects, util.missing_data, ylabel='Mean Proportion of Trials Missing')
plt.plot([600, 600], [0, 1], linestyle = '--', color = 'k')
plt.ylim([0, 1])
plt.figure(7)
plt.title('Proportion of Missing Data over Trial Number')
util.reduce_to_means_over_trial_num(good_subjects, util.missing_data, ylabel='Mean Proportion of Frames Missing')
plt.ylim([0, 1])

# Histogram of number of usable trials per participant
plt.figure(8)
trial_counts_by_subject_shrinky = \
    [len([trial_idx for trial_idx in subject.experiments['shrinky'].trials_to_keep if trial_idx > 0]) \
        for subject in subjects.values() \
            if 'shrinky' in subject.experiments and \
               'eyetrack' in subject.experiments['shrinky'].datatypes and \
                hasattr(subject.experiments['shrinky'], 'trials_to_keep')]
trial_counts_by_subject_noshrinky = \
    [len([trial_idx for trial_idx in subject.experiments['noshrinky'].trials_to_keep if trial_idx > 0]) \
        for subject in subjects.values() \
            if 'noshrinky' in subject.experiments and \
               'eyetrack' in subject.experiments['noshrinky'].datatypes and \
                hasattr(subject.experiments['noshrinky'], 'trials_to_keep')]
plt.subplot(2, 1, 1)
plt.title('Distribution of Usable Trials per Participant')
plt.hist(trial_counts_by_subject_shrinky, bins=25)
plt.xlim((0, 11))
plt.ylabel('Shrinky')
plt.subplot(2, 1, 2)
plt.hist(trial_counts_by_subject_noshrinky, bins=25)
plt.xlim((0, 11))
plt.xlabel('Number of Trials Kept')
plt.ylabel('No Shrinky')

plt.figure(10)
plt.title('Behavioral Performance over Trial Number, Shrinky and Noshrinky')
util.reduce_to_means_over_trial_num(good_subjects, util.behavioral_performance, ylabel='Mean Proportion of Correct Responses')
plt.ylim(bottom=0)

# Scatter plot eyetracking performance over behavioral performance
plt.figure(11)
behavioral_x_func = lambda experiment : np.mean([util.behavioral_performance(*trial) \
                                        for trial in zip(experiment.datatypes['trackit'].trials, experiment.datatypes['eyetrack'].trials)])
                                        # for trackit_trial in experiment.datatypes['trackit'].trials])
HMM_y_func = lambda experiment : np.nanmean([np.nanmean(util.performance_according_to_HMM(*trial)) \
                                        for trial in zip(experiment.datatypes['trackit'].trials, experiment.datatypes['eyetrack'].trials)])
# naive_y_func = lambda experiment : np.nanmean([np.nanmean(util.performance_according_to_naive(*trial)) \
#                                         for trial in zip(experiment.datatypes['trackit'].trials, experiment.datatypes['eyetrack'].trials)])
util.reduce_to_corr_over_experiments(good_subjects,
                                     behavioral_x_func,
                                     [('HMM', HMM_y_func)],# ('Naive', naive_y_func)],
                                     xlabel='Behavioral Performance',
                                     ylabel='Eyetracking Performance')

# Plot of HMM, Naive, and behavioral average performance over age
plt.figure(12, figsize=(6,3))
shrinky_ages = np.array([subject.experiments['shrinky'].age for subject in good_subjects.values()])
noshrinky_ages = np.array([subject.experiments['noshrinky'].age for subject in good_subjects.values()])
shrinky_behavioral = np.array([behavioral_x_func(subject.experiments['shrinky']) for subject in good_subjects.values()])
noshrinky_behavioral = np.array([behavioral_x_func(subject.experiments['noshrinky']) for subject in good_subjects.values()])
shrinky_HMM = np.array([HMM_y_func(subject.experiments['shrinky']) for subject in good_subjects.values()])
noshrinky_HMM = np.array([HMM_y_func(subject.experiments['noshrinky']) for subject in good_subjects.values()])

# shrinky_naive = zip(*[(subject.experiments['shrinky'].age, naive_y_func(subject.experiments['shrinky'])) for subject in good_subjects.values()])
# noshrinky_naive = zip(*[(subject.experiments['noshrinky'].age, naive_y_func(subject.experiments['noshrinky'])) for subject in good_subjects.values()])

plt.subplot(1,2,1)
# Shrinky
plt.scatter(shrinky_ages, shrinky_behavioral, c='C4')
reg = sm.OLS(shrinky_behavioral,sm.add_constant(shrinky_ages)).fit()
print('\nRegression of Shrinky Behavioral Performance over Age:')
print(reg.summary())
pred_input = np.linspace(shrinky_ages.min(),shrinky_ages.max(),1000)
plt.plot(pred_input,reg.predict(sm.add_constant(pred_input)),c='C4', label='Behavioral Exo.', ls='--')
# NoShrinky
plt.scatter(noshrinky_ages, noshrinky_behavioral, c='C5')
pred_input = np.linspace(noshrinky_ages.min(),noshrinky_ages.max(),1000)
reg = sm.OLS(noshrinky_behavioral,sm.add_constant(noshrinky_ages)).fit()
print('\nRegression of NoShrinky Behavioral Performance over Age:')
print(reg.summary())
plt.plot(pred_input,reg.predict(sm.add_constant(pred_input)),c='C5', label='Behavioral Endo.')
plt.legend(loc='lower right')
plt.ylabel('Mean Performance', fontsize=14)
plt.xlabel('Age', fontsize=14)
plt.ylim((0,1))

plt.subplot(1,2,2)
# Shrinky
plt.scatter(shrinky_ages, shrinky_HMM, c='C0')
reg = sm.OLS(shrinky_HMM,sm.add_constant(shrinky_ages)).fit()
print('\nRegression of Shrinky HMM Performance over Age:')
print(reg.summary())
pred_input = np.linspace(shrinky_ages.min(),shrinky_ages.max(),1000)
plt.plot(pred_input,reg.predict(sm.add_constant(pred_input)),c='C0', label='HMM Exogenous', ls='--')
# NoShrinky
plt.scatter(noshrinky_ages, noshrinky_HMM, c='C1')
reg = sm.OLS(noshrinky_HMM,sm.add_constant(noshrinky_ages)).fit()
print('\nRegression of NoShrinky HMM Performance over Age:')
print(reg.summary())
pred_input = np.linspace(noshrinky_ages.min(),noshrinky_ages.max(),1000)
plt.plot(pred_input,reg.predict(sm.add_constant(pred_input)),c='C1', label='HMM Endogenous')
plt.legend(loc='lower right')
plt.xlabel('Age', fontsize=14)
plt.ylim((0,1))
plt.tight_layout()

# plt.subplot(1,2,2)
# plt.scatter(*shrinky_HMM,c='C0',label='HMM Shrinky')
# plt.scatter(*noshrinky_HMM,c='C1',label='HMM Noshrinky')
# plt.plot(np.unique(shrinky_HMM[0]),\
#          np.poly1d(np.polyfit(shrinky_HMM[0],shrinky_HMM[1],1))(np.unique(shrinky_HMM[0])),\
#          c='C0')
# plt.plot(np.unique(noshrinky_HMM[0]),\
#          np.poly1d(np.polyfit(noshrinky_HMM[0],noshrinky_HMM[1],1))(np.unique(noshrinky_HMM[0])),\
#          c='C1')
# plt.legend(loc='lower right')
# plt.xlabel('Participant Age')
# plt.ylim((0,1))

# plt.subplot(1,3,3)
# plt.scatter(*shrinky_naive,c='C2',label='Naive Shrinky')
# plt.scatter(*noshrinky_naive,c='C3',label='Naive Noshrinky')
# plt.plot(np.unique(shrinky_naive[0]),\
#          np.poly1d(np.polyfit(shrinky_naive[0],shrinky_naive[1],1))(np.unique(shrinky_naive[0])),\
#          c='C2')
# plt.plot(np.unique(noshrinky_naive[0]),\
#          np.poly1d(np.polyfit(noshrinky_naive[0],noshrinky_naive[1],1))(np.unique(noshrinky_naive[0])),\
#          c='C3')
# axes = plt.gca()
# axes.set_ylim([0,1])
# plt.legend()
# plt.ylim((0,1))




# print('\nRegression of NoShrinky Behavioral Performance over Age:')
# print(sm.OLS(noshrinky_behavioral[1],sm.add_constant(noshrinky_behavioral[0])).fit().summary())
# print('\nRegression of Shrinky HMM Performance over Age:')
# print(sm.OLS(shrinky_HMM[1],sm.add_constant(shrinky_HMM[0])).fit().summary())
# print('\nRegression of NoShrinky HMM Performance over Age:')
# print(sm.OLS(noshrinky_HMM[1],sm.add_constant(noshrinky_HMM[0])).fit().summary())
# TODO: CHECK WHETHER WE WANT TO INCLUDE THE NAIVE RESULTS?
# print('\nRegression of Shrinky Naive Performance over Age:')
# print(sm.OLS(shrinky_naive[1],sm.add_constant(shrinky_naive[0])).fit().summary())
# print('\nRegression of NoShrinky Naive Performance over Age:')
# print(sm.OLS(noshrinky_naive[1],sm.add_constant(noshrinky_naive[0])).fit().summary())

# plt.figure(13)
# plt.subplot(2,1,1)
# plt.title('Shrinky')
# shrinky_flattened_behavioral = [trackit_trial.trial_metadata['gridClickCorrect'] == 'true' \
#                                     for subject in good_subjects.values() \
#                                     for trackit_trial in subject.experiments['shrinky'].datatypes['trackit'].trials]
# shrinky_flattened_HMM = [np.nanmean(util.performance_according_to_HMM(trackit_trial, eyetrack_trial)) \
#                             for subject in good_subjects.values() \
#                             for (trackit_trial, eyetrack_trial) in zip(subject.experiments['shrinky'].datatypes['trackit'].trials,
#                                                                        subject.experiments['shrinky'].datatypes['eyetrack'].trials)]
# shrinky_flattened_naive = [np.nanmean(util.performance_according_to_naive(trackit_trial, eyetrack_trial)) \
#                             for subject in good_subjects.values() \
#                             for (trackit_trial, eyetrack_trial) in zip(subject.experiments['shrinky'].datatypes['trackit'].trials,
#                                                                        subject.experiments['shrinky'].datatypes['eyetrack'].trials)]
# shrinky_HMM_incorrect_trials = [HMM for (behavioral, HMM) in zip(shrinky_flattened_behavioral, shrinky_flattened_HMM) \
#                                     if not behavioral and not math.isnan(HMM)]
# shrinky_HMM_correct_trials = [HMM for (behavioral, HMM) in zip(shrinky_flattened_behavioral, shrinky_flattened_HMM) \
#                                     if behavioral and not math.isnan(HMM)]
# shrinky_naive_incorrect_trials = [naive for (behavioral, naive) in zip(shrinky_flattened_behavioral, shrinky_flattened_naive)
#                                     if not behavioral and not math.isnan(naive)]
# shrinky_naive_correct_trials = [naive for (behavioral, naive) in zip(shrinky_flattened_behavioral, shrinky_flattened_naive)
#                                     if behavioral and not math.isnan(naive)]
# 
# violin_parts = plt.violinplot([shrinky_naive_incorrect_trials,
#                                shrinky_HMM_incorrect_trials,
#                                shrinky_naive_correct_trials,
#                                shrinky_HMM_correct_trials],
#                         positions=[0,1,5,6],
#                         showmedians=True)
# plt.xlim((-1,7))
# violin_parts['bodies'][0].set_color('r')
# violin_parts['bodies'][2].set_color('r')
# plt.legend([mpatches.Patch(color='red', alpha=0.4), mpatches.Patch(color='blue', alpha=0.25)], ['Naive', 'HMM'], loc='upper center')
# for partname in ('cbars','cmins','cmaxes','cmeans','cmedians'):
#     try:
#       vp = violin_parts[partname]
#       vp.set_edgecolor('black')
#     except KeyError:
#       pass
# plt.ylabel('Proportion of trial on target')
# plt.xticks([0.5, 5.5], ('Incorrect Trials', 'Correct Trials'))
# 
# plt.subplot(2,1,2)
# plt.title('No Shrinky')
# noshrinky_flattened_behavioral = [trackit_trial.trial_metadata['gridClickCorrect'] == 'true' \
#                                     for subject in good_subjects.values() \
#                                     for trackit_trial in subject.experiments['noshrinky'].datatypes['trackit'].trials]
# noshrinky_flattened_HMM = [np.nanmean(util.performance_according_to_HMM(trackit_trial, eyetrack_trial)) \
#                             for subject in good_subjects.values() \
#                             for (trackit_trial, eyetrack_trial) in zip(subject.experiments['noshrinky'].datatypes['trackit'].trials,
#                                                                        subject.experiments['noshrinky'].datatypes['eyetrack'].trials)]
# noshrinky_flattened_naive = [np.nanmean(util.performance_according_to_naive(trackit_trial, eyetrack_trial)) \
#                             for subject in good_subjects.values() \
#                             for (trackit_trial, eyetrack_trial) in zip(subject.experiments['noshrinky'].datatypes['trackit'].trials,
#                                                                        subject.experiments['noshrinky'].datatypes['eyetrack'].trials)]
# noshrinky_HMM_incorrect_trials = [HMM for (behavioral, HMM) in zip(noshrinky_flattened_behavioral, noshrinky_flattened_HMM) \
#                                     if not behavioral and not math.isnan(HMM)]
# noshrinky_HMM_correct_trials = [HMM for (behavioral, HMM) in zip(noshrinky_flattened_behavioral, noshrinky_flattened_HMM) \
#                                     if behavioral and not math.isnan(HMM)]
# noshrinky_naive_incorrect_trials = [naive for (behavioral, naive) in zip(noshrinky_flattened_behavioral, noshrinky_flattened_naive)
#                                     if not behavioral and not math.isnan(naive)]
# noshrinky_naive_correct_trials = [naive for (behavioral, naive) in zip(noshrinky_flattened_behavioral, noshrinky_flattened_naive)
#                                     if behavioral and not math.isnan(naive)]
# 
# violin_parts = plt.violinplot([noshrinky_naive_incorrect_trials,
#                                noshrinky_HMM_incorrect_trials,
#                                noshrinky_naive_correct_trials,
#                                noshrinky_HMM_correct_trials],
#                         positions=[0,1,5,6],
#                         showmedians=True)
# plt.xlim((-1,7))
# violin_parts['bodies'][0].set_color('r')
# violin_parts['bodies'][2].set_color('r')
# plt.legend([mpatches.Patch(color='red', alpha=0.4), mpatches.Patch(color='blue', alpha=0.25)], ['Naive', 'HMM'], loc='upper center')
# for partname in ('cbars','cmins','cmaxes','cmeans','cmedians'):
#     try:
#       vp = violin_parts[partname]
#       vp.set_edgecolor('black')
#     except KeyError:
#       pass
# plt.ylabel('Proportion of trial on target')
# plt.xticks([0.5, 5.5], ('Incorrect Trials', 'Correct Trials'))
# 
# plt.figure(14)
# util.within_subjects_diff_means_over_trial_time(good_subjects, [('HMM', util.performance_according_to_HMM),('Naive', util.performance_according_to_naive)], shrinky_minus_noshrinky=True)
# plt.plot([600, 600], [-1,1], linestyle = '--', color = 'k')

# Within-subjects (Shrinky - Noshrinky) difference
shrinky_performance_by_subject = []
noshrinky_performance_by_subject = []
for subject in good_subjects.values():
  subject_performances = []
  for (trackit_trial, eyetrack_trial) in zip(subject.experiments['shrinky'].datatypes['trackit'].trials, subject.experiments['shrinky'].datatypes['eyetrack'].trials):
    subject_performances.append(np.nanmean(util.performance_according_to_HMM(trackit_trial, eyetrack_trial)))
  shrinky_performance_by_subject.append(np.nanmean(subject_performances))

  subject_performances = []
  for (trackit_trial, eyetrack_trial) in zip(subject.experiments['noshrinky'].datatypes['trackit'].trials, subject.experiments['noshrinky'].datatypes['eyetrack'].trials):
    subject_performances.append(np.nanmean(util.performance_according_to_HMM(trackit_trial, eyetrack_trial)))
  noshrinky_performance_by_subject.append(np.nanmean(subject_performances))
diffs = np.array(shrinky_performance_by_subject) - np.array(noshrinky_performance_by_subject)
print('Within-Subjects Shrinky - NoShrinky difference:', '  Mean:', np.nanmean(diffs), '  95% CI:', 1.96*np.std(diffs)/math.sqrt(len(diffs)))
print('Between-Subjects Two-Sample t-test:', stats.ttest_ind(shrinky_performance_by_subject, noshrinky_performance_by_subject))

plt.figure(15)
all_changepoint_lengths_shrinky = []
all_changepoint_lengths_noshrinky = []
for subject in good_subjects.values():
  subject_changepoint_lengths = []
  for trial in subject.experiments['shrinky'].datatypes['eyetrack'].trials:
    trial_changepoints = [0]
    trial_changepoints.extend(np.where(trial.HMM_MLE[:-1] != trial.HMM_MLE[1:])[0])
    trial_changepoints = np.array(trial_changepoints)
    if len(trial_changepoints) > 1:
      trial_changepoint_lengths = np.log(1 + trial_changepoints[1:] - trial_changepoints[:-1])
      subject_changepoint_lengths.extend(trial_changepoint_lengths)
  all_changepoint_lengths_shrinky.extend(subject_changepoint_lengths)

  subject_changepoint_lengths = []
  for trial in subject.experiments['noshrinky'].datatypes['eyetrack'].trials:
    trial_changepoints = [0]
    trial_changepoints.extend(np.where(trial.HMM_MLE[:-1] != trial.HMM_MLE[1:])[0])
    trial_changepoints = np.array(trial_changepoints)
    if len(trial_changepoints) > 1:
      trial_changepoint_lengths = np.log(1 + trial_changepoints[1:] - trial_changepoints[:-1])
      subject_changepoint_lengths.extend(trial_changepoint_lengths)
  all_changepoint_lengths_noshrinky.extend(subject_changepoint_lengths)

plt.subplot(2,1,1)
plt.hist(all_changepoint_lengths_shrinky, label='Shrinky')
plt.subplot(2,1,2)
plt.hist(all_changepoint_lengths_noshrinky, label='NoShrinky')

# If data weren't already cached, cache them
if not data_cached:
  with open(cache_file , 'w') as f:
    pickle.dump(subjects, f)
  print('Cached subjects to file \''  + cache_file + '\'')
  data_cached = True

plt.show()
