import matplotlib.pyplot as plt
import pickle
import numpy as np
import csv, math
from sklearn.metrics import confusion_matrix

import eyetracking2_util as util
import naive_eyetracking
from eyetracking_hmm import compute_likelihood

np.set_printoptions(suppress=True)

# BEGIN CODE FOR COMPARING HMM AND NAIVE TO HUMAN CODING

data_root = 'data/shrinky++shrinky/human_coded/'
# These sessions have to be redone:
# ('coder2', 'AF7', 'shrinky'),
sessions_to_use = [('coder1', 'A272', 'noshrinky'),
                   ('coder1', 'A272', 'shrinky'),
                   ('coder1', 'A276', 'noshrinky'),
                   ('coder1', 'A276', 'shrinky'),
                   ('coder1', 'A294', 'noshrinky'),
                   ('coder1', 'A297', 'shrinky'),
                   ('coder1', 'A298', 'noshrinky'),
                   ('coder1', 'A298', 'shrinky'),
                   ('coder1', 'A326', 'noshrinky'),
                   ('coder1', 'A327', 'noshrinky'),
                   ('coder1', 'A327', 'shrinky'),
                   ('coder1', 'A338', 'noshrinky'),
                   ('coder1', 'A338', 'shrinky'),
                   ('coder1', 'A339', 'noshrinky'),
                   ('coder1', 'A340', 'noshrinky'),
                   ('coder1', 'A340', 'shrinky'),
                   ('coder1', 'A345', 'shrinky'),
                   # ('coder1', 'A380', 'noshrinky'),
                   # ('coder1', 'A380', 'shrinky'),
                   ('coder1', 'AF11', 'noshrinky'),
                   ('coder1', 'AF11', 'shrinky'),
                   ('coder1', 'AF2',  'noshrinky'),
                   ('coder1', 'AF2',  'shrinky'),
                   ('coder1', 'B37',  'noshrinky'),
                   ('coder1', 'B37',  'shrinky'),
                   # ('coder1', 'B56',  'noshrinky'),
                   # ('coder1', 'C59',  'noshrinky'),
                   ('coder1', 'E22',  'noshrinky'),
                   ('coder1', 'E22',  'shrinky'),
                   ('coder1', 'E29',  'noshrinky'),
                   ('coder1', 'E29',  'shrinky'),
                   ('coder1', 'E3',   'noshrinky'),
                   ('coder1', 'E3',   'shrinky'),
                   ('coder1', 'E51',  'noshrinky'),
                   ('coder1', 'E51',  'shrinky'),
                   ('coder1', 'E52',  'noshrinky'),
                   ('coder1', 'E52',  'shrinky'),
                   # ('coder1', 'L177', 'noshrinky'),
                   # ('coder1', 'L177', 'shrinky'),
                   ('coder1', 'M140', 'noshrinky'),
                   ('coder1', 'M143', 'noshrinky'),
                   ('coder1', 'M143', 'shrinky'),
                   ('coder1', 'M146', 'noshrinky'),
                   ('coder1', 'M146', 'shrinky'),
                   # ('coder1', 'M191', 'noshrinky'),
                   # ('coder1', 'M191', 'shrinky'),
                   # ('coder1', 'M197', 'noshrinky'),
                   # ('coder1', 'M197', 'shrinky'),
                   # ('coder1', 'M198', 'noshrinky'),
                   # ('coder1', 'M198', 'shrinky'),
                   # ('coder1', 'M201', 'shrinky'),
                   # ('coder1', 'M206', 'noshrinky'),
                   ('coder1', 'R12',  'shrinky'),
                   ('coder1', 'R34',  'noshrinky'),
                   ('coder1', 'R34',  'shrinky'),
                   ('coder1', 'R35',  'noshrinky'),
                   ('coder1', 'R35',  'shrinky'),
                   ('coder1', 'R37',  'noshrinky'),
                   ('coder1', 'R37',  'shrinky'),
                   ('coder3', 'A276', 'shrinky'),
                   ('coder3', 'A294', 'noshrinky'),
                   ('coder3', 'A295', 'noshrinky'),
                   ('coder3', 'A297', 'noshrinky'),
                   ('coder3', 'A297', 'shrinky'),
                   ('coder3', 'A298', 'shrinky'),
                   ('coder3', 'A326', 'shrinky'),
                   ('coder3', 'A345', 'noshrinky'),
                   # ('coder3', 'A380', 'noshrinky'),
                   ('coder3', 'AF11', 'noshrinky'),
                   ('coder3', 'AF2',  'noshrinky'),
                   ('coder3', 'AF7',  'noshrinky'),
                   ('coder3', 'AF8',  'noshrinky'),
                   ('coder3', 'AF8',  'shrinky'),
                   ('coder3', 'B33',  'noshrinky'),
                   ('coder3', 'B33',  'shrinky'),
                   ('coder3', 'B38',  'noshrinky'),
                   ('coder3', 'B38',  'shrinky'),
                   # ('coder3', 'B56',  'shrinky'),
                   # ('coder3', 'C59',  'shrinky'),
                   ('coder3', 'E22',  'shrinky'),
                   ('coder3', 'E29',  'noshrinky'),
                   ('coder3', 'E3',   'noshrinky'),
                   ('coder3', 'E51',  'shrinky'),
                   ('coder3', 'M140', 'shrinky'),
                   ('coder3', 'M146', 'shrinky'),
                   # ('coder3', 'M193', 'noshrinky'),
                   # ('coder3', 'M193', 'shrinky'),
                   # ('coder3', 'M198', 'shrinky'),
                   # ('coder3', 'M201', 'noshrinky'),
                   # ('coder3', 'M206', 'noshrinky'),
                   # ('coder3', 'M206', 'shrinky'),
                   ('coder3', 'R12',  'noshrinky'),
                   ('coder3', 'R17',  'noshrinky'),
                   ('coder3', 'R17',  'shrinky'),
                   ('coder3', 'R34',  'noshrinky'),
                   ('coder3', 'R34',  'shrinky'),
                   ('coder3', 'R35',  'shrinky'),
                   ('coder3', 'R37',  'noshrinky')]

# Range of sigma values over which to plot results.
sigmas = np.array(range(25,400,25) + range(400,1100,100))

# Temporal slack for computing confusion matrices. Higher gives more lenient classification results.
# The paper presents results for num_slack_frames = 0 and num_slack_frames = 1.
num_slack_frames = 0

# Allocate space for all statistics we collect
HMM_acc_means = np.zeros((len(sigmas),))
naive_acc_means = np.zeros((len(sigmas),))
HMM_acc_stes = np.zeros((len(sigmas),))
naive_acc_stes = np.zeros((len(sigmas),))
HMM_prec_means = np.zeros((len(sigmas),))
naive_prec_means = np.zeros((len(sigmas),))
HMM_prec_stes = np.zeros((len(sigmas),))
naive_prec_stes = np.zeros((len(sigmas),))
HMM_rec_means = np.zeros((len(sigmas),))
naive_rec_means = np.zeros((len(sigmas),))
HMM_rec_stes = np.zeros((len(sigmas),))
naive_rec_stes = np.zeros((len(sigmas),))
HMM_F1_means = np.zeros((len(sigmas),))
naive_F1_means = np.zeros((len(sigmas),))
HMM_F1_stes = np.zeros((len(sigmas),))
naive_F1_stes = np.zeros((len(sigmas),))
HMM_MCC_means = np.zeros((len(sigmas),))
naive_MCC_means = np.zeros((len(sigmas),))
HMM_MCC_stes = np.zeros((len(sigmas),))
naive_MCC_stes = np.zeros((len(sigmas),))
HMM_likelihood_means = np.zeros((len(sigmas),))
HMM_likelihood_stes = np.zeros((len(sigmas),))

for (sigma_idx, sigma) in enumerate(sigmas):
  fname = 'cache/old/cache_sigma' + str(sigma)
  print('Loading file ' + fname + '...')
  with open(fname, 'r') as f:
    subjects = pickle.load(f)

  HMM_prop_correct_by_session = np.zeros((len(sessions_to_use),))
  naive_prop_correct_by_session = np.zeros((len(sessions_to_use),))
  HMM_precision_by_session = np.zeros((len(sessions_to_use),))
  naive_precision_by_session = np.zeros((len(sessions_to_use),))
  HMM_recall_by_session = np.zeros((len(sessions_to_use),))
  naive_recall_by_session = np.zeros((len(sessions_to_use),))
  HMM_F1_by_session = np.zeros((len(sessions_to_use),))
  naive_F1_by_session = np.zeros((len(sessions_to_use),))
  HMM_MCC_by_session = np.zeros((len(sessions_to_use),))
  naive_MCC_by_session = np.zeros((len(sessions_to_use),))
  HMM_likelihood_by_session = np.zeros((len(sessions_to_use),))
  total_confusion_matrix_HMM = np.zeros((2,2))
  total_confusion_matrix_naive = np.zeros((2,2))
  for (session_idx, (coder, subject_ID, experiment)) in enumerate(sessions_to_use):

    trials_to_show = range(int(subject.experiments[experiment].datatypes['trackit'].metadata['Trial Count']))[1:]
    session_HMM_prop_correct_by_frame = []
    session_naive_prop_correct_by_frame = []
    confusion_matrix_HMM = np.zeros((2,2))
    confusion_matrix_naive = np.zeros((2,2))

    session_changepoints_HMM = []
    session_changepoints_naive = []
    session_changepoints_human = []
    session_likelihoods = []
    for trial_idx in trials_to_show:
      trackit_trial = subject.experiments[experiment].datatypes['trackit'].trials[trial_idx]
      eyetrack_trial = subject.experiments[experiment].datatypes['eyetrack'].trials[trial_idx]
      eyetrack = eyetrack_trial.data[:, 1:]

      # Compute likelihood of HMM predicted state sequence under HMM model
      session_likelihoods.append(compute_likelihood(eyetrack, trackit_trial.object_positions, eyetrack_trial.HMM_MLE, sigma**2))
      
      # Extract cached HMM maximum likelihood estimate, subsampled by a factor of 6, to compare with human coding
      trial_HMM = eyetrack_trial.HMM_MLE[::6]

      # Calculate naive maximum likelihood estimate
      # naive_eyetracking.get_trackit_MLE() expects the target and distractor positions as separate arguments
      target = trackit_trial.object_positions[trackit_trial.target_index,:,:]
      # NOTE: This works because np.delete is not in-place; it makes a copy:
      distractors = np.delete(trackit_trial.object_positions, trackit_trial.target_index, axis=0)
      trial_naive = naive_eyetracking.get_trackit_MLE(np.swapaxes(eyetrack,0,1), np.swapaxes(target,0,1), np.swapaxes(distractors,1,2))[::6]

      human_coding_filename = data_root + coder + '/' + subject_ID + '_' + experiment + '_trial_' + str(trial_idx) + '_coding.csv'
      trial_human = []
      with open(human_coding_filename, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader, None) # Skip CSV header line
        for line in reader:
          if line[1] in ['Off Screen', 'Off Task', '']: # Ignore these frames, since they don't correspond to decodable states
            trial_human.append(-1)
          else:
            trial_human.append(int(line[1][-1:])) # NOTE: This breaks if there are >10 total objects!
          # trial_human.append(line[1] == 'Obect 0')
      trial_human = np.array(trial_human)

      # HMM and Naive predictions should always be the same number of frames.
      assert trial_HMM.shape[0] == trial_naive.shape[0]
      # HMM and human codings should always be within 2 frames of each other.
      if abs(trial_HMM.shape[0] - trial_human.shape[0]) > 2:
        print('Coder: ' + coder + ' Subject: ' + subject_ID + ' Trial: ' + str(trial_idx) + ' len(HMM): ' + str(trial_HMM.shape[0]) + ' len(human): ' + str(trial_human.shape[0]))

      # Truncate to the shorter of the HMM/naive and human coding lengths
      if trial_HMM.shape[0] < trial_human.shape[0]:
        trial_human = trial_human[:trial_HMM.shape[0]]
      elif trial_HMM.shape[0] > trial_human.shape[0]:
        trial_HMM = trial_HMM[:trial_human.shape[0]]
        trial_naive = trial_naive[:trial_human.shape[0]]

      # Remove frames where human coding is `Off Screen'
      trial_HMM = trial_HMM[trial_human >= 0]
      trial_naive = trial_naive[trial_human >= 0]
      trial_human = trial_human[trial_human >= 0]

      # Concatenate accuracies for current trial
      session_HMM_prop_correct_by_frame.extend(trial_HMM == trial_human)
      session_naive_prop_correct_by_frame.extend(trial_naive == trial_human)

      # Add confusion matrix changepoints for current trial
      if len(trial_HMM) > 0:
        changepoints_HMM = util.get_changepoints(trial_HMM)
        changepoints_naive = util.get_changepoints(trial_naive)
        changepoints_human = util.get_changepoints(trial_human)
        confusion_matrix_HMM += util.generalized_2x2_confusion_matrix(changepoints_HMM, changepoints_human, max_dist=num_slack_frames)
        confusion_matrix_naive += util.generalized_2x2_confusion_matrix(changepoints_naive, changepoints_human, max_dist=num_slack_frames)

      total_confusion_matrix_HMM += confusion_matrix_HMM
      total_confusion_matrix_naive += confusion_matrix_naive

    # Aggregate over frames or trials within session
    HMM_prop_correct_by_session[session_idx] = np.mean(session_HMM_prop_correct_by_frame)
    naive_prop_correct_by_session[session_idx] = np.mean(session_naive_prop_correct_by_frame)
    HMM_precision_by_session[session_idx], HMM_recall_by_session[session_idx], HMM_F1_by_session[session_idx], HMM_MCC_by_session[session_idx] = util.classification_performance(confusion_matrix_HMM)
    HMM_likelihood_by_session[session_idx] = np.nanmean(session_likelihoods)

    naive_precision_by_session[session_idx], naive_recall_by_session[session_idx], naive_F1_by_session[session_idx], naive_MCC_by_session[session_idx] = util.classification_performance(confusion_matrix_naive)

  # Aggregate over sessions
  HMM_acc_means[sigma_idx] = np.nanmean(HMM_prop_correct_by_session)
  HMM_acc_stes[sigma_idx] = 1.96*np.nanstd(HMM_prop_correct_by_session)/math.sqrt(len(sessions_to_use))
  naive_acc_means[sigma_idx] = np.nanmean(naive_prop_correct_by_session)
  naive_acc_stes[sigma_idx] = 1.96*np.nanstd(naive_prop_correct_by_session)/math.sqrt(len(sessions_to_use))
  HMM_prec_means[sigma_idx] = np.nanmean(HMM_precision_by_session)
  HMM_prec_stes[sigma_idx] = 1.96*np.nanstd(HMM_precision_by_session)/math.sqrt(len(sessions_to_use))
  naive_prec_means[sigma_idx] = np.nanmean(naive_precision_by_session)
  naive_prec_stes[sigma_idx] = 1.96*np.nanstd(naive_precision_by_session)/math.sqrt(len(sessions_to_use))
  HMM_rec_means[sigma_idx] = np.nanmean(HMM_recall_by_session)
  HMM_rec_stes[sigma_idx] = 1.96*np.nanstd(HMM_recall_by_session)/math.sqrt(len(sessions_to_use))
  naive_rec_means[sigma_idx] = np.nanmean(naive_recall_by_session)
  naive_rec_stes[sigma_idx] = 1.96*np.nanstd(naive_recall_by_session)/math.sqrt(len(sessions_to_use))
  HMM_F1_means[sigma_idx] = np.nanmean(HMM_F1_by_session)
  HMM_F1_stes[sigma_idx] = 1.96*np.nanstd(HMM_F1_by_session)/math.sqrt(len(sessions_to_use))
  naive_F1_means[sigma_idx] = np.nanmean(naive_F1_by_session)
  naive_F1_stes[sigma_idx] = 1.96*np.nanstd(naive_F1_by_session)/math.sqrt(len(sessions_to_use))
  HMM_MCC_means[sigma_idx] = np.nanmean(HMM_MCC_by_session)
  HMM_MCC_stes[sigma_idx] = 1.96*np.nanstd(HMM_MCC_by_session)/math.sqrt(len(sessions_to_use))
  naive_MCC_means[sigma_idx] = np.nanmean(naive_MCC_by_session)
  naive_MCC_stes[sigma_idx] = 1.96*np.nanstd(naive_MCC_by_session)/math.sqrt(len(sessions_to_use))
  HMM_likelihood_means[sigma_idx] = np.nanmean(HMM_likelihood_by_session[session_idx])
  HMM_likelihood_stes[sigma_idx] = 1.96*np.nanstd(HMM_likelihood_by_session)/math.sqrt(len(sessions_to_use))

  print('HMM\n', total_confusion_matrix_HMM)
  print('naive\n', total_confusion_matrix_naive)

print('HMM Accuracy', HMM_acc_means, HMM_acc_stes)
print('Naive Accuracy', naive_acc_means, naive_acc_stes)
print('HMM Precision', HMM_prec_means, HMM_prec_stes)
print('Naive Precision', naive_prec_means, naive_prec_stes)
print('HMM Recall', HMM_rec_means, HMM_rec_stes)
print('Naive Recall', naive_rec_means, naive_rec_stes)
print('HMM F1 Score', HMM_F1_means, HMM_F1_stes)
print('Naive F1 Score', naive_F1_means, naive_F1_stes)
print('HMM MCC', HMM_MCC_means, HMM_MCC_stes)
print('Naive MCC', naive_MCC_means, naive_MCC_stes)

# Indexed by num_slack_frames and then by measure of performance (from interrater_reliability.py)
# Number of Frames used to measure Inter-rater reliability: 25075
# This first set of results omits frames classified as Off Task by either coder (dotted black line in plots)
interrater_switch_performance_no_off_task = [{ 'prec/rec/F1' : 0.553980678372, 'MCC' : 0.540290693059 },
                                             { 'prec/rec/F1' : 0.757650045421, 'MCC' : 0.750207642797 }]
interrater_switch_performance_95CI_no_off_task = [{ 'prec/rec/F1' : 0.0541262216345, 'MCC' : 0.0542673928783 },
                                                  { 'prec/rec/F1' : 0.0466594230506, 'MCC' : 0.0471372108679 }]
interrater_average_no_off_task = 0.9556202255365588
interrater_average_95CI_no_off_task = 0.0224242939696

# Inter-rater results including frames classified as Off Task (dotted white line in plots)
interrater_switch_performance = [{ 'prec/rec/F1' : 0.219524334621, 'MCC' : 0.172656208336 },
                                 { 'prec/rec/F1' : 0.563307726573, 'MCC' : 0.536115618573 }]
interrater_switch_performance_95CI = [{ 'prec/rec/F1' : 0.045071766976, 'MCC' : 0.0411545324437 },
                                      { 'prec/rec/F1' : 0.0540062688792, 'MCC' : 0.054302230802 }]
interrater_average = 0.8500231803430691
interrater_average_95CI = 0.0388786360775

# Generate plot of classification accuracy
plt.figure()
plt.plot(sigmas, HMM_acc_means, label='HMM', marker='.')
plt.fill_between(sigmas, HMM_acc_means - HMM_acc_stes, HMM_acc_means + HMM_acc_stes, facecolor='C0', interpolate=True, alpha=0.3)
plt.plot(sigmas, naive_acc_means, label='Naive')
plt.fill_between(sigmas, naive_acc_means - naive_acc_stes, naive_acc_means + naive_acc_stes, facecolor='C1', interpolate=True, alpha=0.3)
human_to_plot_upper = interrater_average_no_off_task*np.array([1.0, 1.0])
human_to_plot_lower = interrater_average*np.array([1.0, 1.0])
plt.plot([sigmas.min(), sigmas.max()], human_to_plot_upper, ls='--', c='k')
plt.plot([sigmas.min(), sigmas.max()], human_to_plot_lower, ls='--', c='w')
plt.fill_between([sigmas.min(), sigmas.max()], human_to_plot_lower - interrater_average_95CI, human_to_plot_upper + interrater_average_95CI_no_off_task, facecolor='k', interpolate=True, alpha=0.3, label='Human (Inter-rater)')
plt.ylabel('Agreement with Human', fontsize=14)
plt.xlabel(r'$\sigma$' + ' parameter (pixels)', fontsize=14)
plt.ylim(bottom = 0.5, top = 1.0)
plt.legend(loc='best', fancybox=True, framealpha=0.5)

# Generate plot of switch classification Precision
plt.figure()
plt.subplot(2,2,1)
plt.plot(sigmas, HMM_prec_means, label='HMM', marker='.')
plt.fill_between(sigmas, HMM_prec_means - HMM_prec_stes, HMM_prec_means + HMM_prec_stes, facecolor='C0', interpolate=True, alpha=0.3)
plt.plot(sigmas, naive_prec_means, label='Naive')
plt.fill_between(sigmas, naive_prec_means - naive_prec_stes, naive_prec_means + naive_prec_stes, facecolor='C1', interpolate=True, alpha=0.3)
human_to_plot_upper = interrater_switch_performance_no_off_task[num_slack_frames]['prec/rec/F1']*np.array([1.0, 1.0])
human_to_plot_lower = interrater_switch_performance[num_slack_frames]['prec/rec/F1']*np.array([1.0, 1.0])
plt.plot([sigmas.min(), sigmas.max()], human_to_plot_upper, ls='--', c='k')
plt.plot([sigmas.min(), sigmas.max()], human_to_plot_lower, ls='--', c='w')
plt.fill_between([sigmas.min(), sigmas.max()], human_to_plot_lower - interrater_switch_performance_95CI[num_slack_frames]['prec/rec/F1'], human_to_plot_upper + interrater_switch_performance_95CI_no_off_task[num_slack_frames]['prec/rec/F1'], facecolor='k', interpolate=True, alpha=0.3, label='Human (Inter-rater)')
plt.ylabel('Precision', fontsize=14)
plt.ylim(bottom = 0.0, top = 1.0)

# Generate plot of switch classification F1 Score
plt.subplot(2,2,4)
human_to_plot_upper = interrater_switch_performance_no_off_task[num_slack_frames]['prec/rec/F1']*np.array([1.0, 1.0])
human_to_plot_lower = interrater_switch_performance[num_slack_frames]['prec/rec/F1']*np.array([1.0, 1.0])
plt.plot([sigmas.min(), sigmas.max()], human_to_plot_upper, ls='--', c='k')
plt.plot([sigmas.min(), sigmas.max()], human_to_plot_lower, ls='--', c='w')
plt.fill_between([sigmas.min(), sigmas.max()], human_to_plot_lower - interrater_switch_performance_95CI[num_slack_frames]['prec/rec/F1'], human_to_plot_upper + interrater_switch_performance_95CI_no_off_task[num_slack_frames]['prec/rec/F1'], facecolor='k', interpolate=True, alpha=0.3, label='Human (Inter-rater)')
plt.plot(sigmas, HMM_F1_means, label='HMM', marker='.')
plt.fill_between(sigmas, HMM_F1_means - HMM_F1_stes, HMM_F1_means + HMM_F1_stes, facecolor='C0', interpolate=True, alpha=0.3)
plt.plot(sigmas, naive_F1_means, label='Naive')
plt.fill_between(sigmas, naive_F1_means - naive_F1_stes, naive_F1_means + naive_F1_stes, facecolor='C1', interpolate=True, alpha=0.3)
plt.ylabel('F1', fontsize=14)
plt.xlabel(r'$\sigma$' + ' parameter (pixels)', fontsize=14)
plt.ylim(bottom = 0.0, top = 1.0)
plt.legend(loc='best', fancybox=True, framealpha=0.5)

# Generate plot of switch classification Matthew's Correlation Coefficient
plt.subplot(2,2,3)
plt.plot(sigmas, HMM_MCC_means, label='HMM', marker='.')
plt.fill_between(sigmas, HMM_MCC_means - HMM_MCC_stes, HMM_MCC_means + HMM_MCC_stes, facecolor='C0', interpolate=True, alpha=0.3)
plt.plot(sigmas, naive_MCC_means, label='Naive')
plt.fill_between(sigmas, naive_MCC_means - naive_MCC_stes, naive_MCC_means + naive_MCC_stes, facecolor='C1', interpolate=True, alpha=0.3)
human_to_plot_upper = interrater_switch_performance_no_off_task[num_slack_frames]['MCC']*np.array([1.0, 1.0])
human_to_plot_lower = interrater_switch_performance[num_slack_frames]['MCC']*np.array([1.0, 1.0])
plt.plot([sigmas.min(), sigmas.max()], human_to_plot_upper, ls='--', c='k')
plt.plot([sigmas.min(), sigmas.max()], human_to_plot_lower, ls='--', c='w')
plt.fill_between([sigmas.min(), sigmas.max()], human_to_plot_lower - interrater_switch_performance_95CI[num_slack_frames]['MCC'], human_to_plot_upper + interrater_switch_performance_95CI_no_off_task[num_slack_frames]['MCC'], facecolor='k', interpolate=True, alpha=0.3, label='Human (Inter-rater)')
plt.ylabel('MCC', fontsize=14)
plt.xlabel(r'$\sigma$' + ' parameter (pixels)', fontsize=14)
plt.ylim(bottom = 0.0, top = 1.0)

# Generate plot of switch classification Recall
plt.subplot(2,2,2)
plt.plot(sigmas, HMM_rec_means, label='HMM', marker='.')
plt.fill_between(sigmas, HMM_rec_means - HMM_rec_stes, HMM_rec_means + HMM_rec_stes, facecolor='C0', interpolate=True, alpha=0.3)
plt.plot(sigmas, naive_rec_means, label='Naive')
plt.fill_between(sigmas, naive_rec_means - naive_rec_stes, naive_rec_means + naive_rec_stes, facecolor='C1', interpolate=True, alpha=0.3)
human_to_plot_upper = interrater_switch_performance_no_off_task[num_slack_frames]['prec/rec/F1']*np.array([1.0, 1.0])
human_to_plot_lower = interrater_switch_performance[num_slack_frames]['prec/rec/F1']*np.array([1.0, 1.0])
plt.plot([sigmas.min(), sigmas.max()], human_to_plot_upper, ls='--', c='k')
plt.plot([sigmas.min(), sigmas.max()], human_to_plot_lower, ls='--', c='w')
plt.fill_between([sigmas.min(), sigmas.max()], human_to_plot_lower - interrater_switch_performance_95CI[num_slack_frames]['prec/rec/F1'], human_to_plot_upper + interrater_switch_performance_95CI_no_off_task[num_slack_frames]['prec/rec/F1'], facecolor='k', interpolate=True, alpha=0.3, label='Human (Inter-rater)')
plt.ylabel('Recall', fontsize=14)
plt.ylim(bottom = 0.0, top = 1.0)

# Generate plot of likelihood
plt.figure()
plt.plot(sigmas, HMM_likelihood_means, marker='.')
plt.fill_between(sigmas, HMM_likelihood_means - HMM_likelihood_stes, HMM_likelihood_means + HMM_likelihood_stes, facecolor='C0', interpolate=True, alpha=0.3)
plt.ylabel('Log-Likelihood', fontsize=14)
plt.xlabel(r'$\sigma$' + ' parameter (pixels)', fontsize=14)
plt.show()
