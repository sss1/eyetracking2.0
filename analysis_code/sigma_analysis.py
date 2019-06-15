import matplotlib.pyplot as plt
import pickle
import numpy as np
import csv, math

from util import get_changepoints, classification_performance, generalized_2x2_confusion_matrix
import naive_eyetracking

np.set_printoptions(suppress=True)

sessions_to_use = [('coder1', '0',  'noshrinky'),
                   ('coder1', '0',  'shrinky'  ),
                   ('coder1', '31', 'noshrinky'),
                   ('coder1', '31', 'shrinky'  ),
                   ('coder1', '7',  'noshrinky'),
                   ('coder1', '7',  'shrinky'  ),
                   ('coder1', '27', 'noshrinky'),
                   ('coder1', '15', 'shrinky'  ),
                   ('coder1', '28', 'noshrinky'),
                   ('coder1', '28', 'shrinky'  ),
                   ('coder1', '47', 'noshrinky'),
                   ('coder1', '46', 'noshrinky'),
                   ('coder1', '46', 'shrinky'  ),
                   ('coder1', '26', 'noshrinky'),
                   ('coder1', '26', 'shrinky'  ),
                   ('coder1', '30', 'noshrinky'),
                   ('coder1', '18', 'noshrinky'),
                   ('coder1', '18', 'shrinky'  ),
                   ('coder1', '35', 'shrinky'  ),
                   ('coder1', '19', 'noshrinky'),
                   ('coder1', '19', 'shrinky'  ),
                   ('coder1', '3',  'noshrinky'),
                   ('coder1', '3',  'shrinky'  ),
                   ('coder1', '48', 'noshrinky'),
                   ('coder1', '48', 'shrinky'  ),
                   ('coder1', '45', 'noshrinky'),
                   ('coder1', '45', 'shrinky'  ),
                   ('coder1', '1',  'noshrinky'),
                   ('coder1', '6',  'noshrinky'),
                   ('coder1', '41', 'noshrinky'),
                   ('coder1', '41', 'shrinky'  ),
                   ('coder1', '10', 'noshrinky'),
                   ('coder1', '10', 'shrinky'  ),
                   ('coder1', '9',  'noshrinky'),
                   ('coder1', '9',  'shrinky'  ),
                   ('coder1', '38', 'noshrinky'),
                   ('coder1', '38', 'shrinky'  ),
                   ('coder1', '25', 'noshrinky'),
                   ('coder1', '25', 'shrinky'  ),
                   ('coder1', '21', 'noshrinky'),
                   ('coder1', '21', 'shrinky'  ),
                   ('coder1', '20', 'noshrinky'),
                   ('coder1', '33', 'noshrinky'),
                   ('coder1', '33', 'shrinky'  ),
                   ('coder1', '11', 'noshrinky'),
                   ('coder1', '11', 'shrinky'  ),
                   ('coder1', '49', 'noshrinky'),
                   ('coder1', '49', 'shrinky'  ),
                   ('coder1', '13', 'noshrinky'),
                   ('coder1', '13', 'shrinky'  ),
                   ('coder1', '32', 'noshrinky'),
                   ('coder1', '32', 'shrinky'  ),
                   ('coder1', '24', 'shrinky'  ),
                   ('coder1', '2',  'noshrinky'),
                   ('coder1', '37', 'shrinky'  ),
                   ('coder1', '17', 'noshrinky'),
                   ('coder1', '17', 'shrinky'  ),
                   ('coder1', '22', 'noshrinky'),
                   ('coder1', '22', 'shrinky'  ),
                   ('coder3', '0',  'shrinky'  ),
                   ('coder3', '7',  'shrinky'  ),
                   ('coder3', '27', 'noshrinky'),
                   ('coder3', '42', 'noshrinky'),
                   ('coder3', '15', 'noshrinky'),
                   ('coder3', '15', 'shrinky'  ),
                   ('coder3', '28', 'shrinky'  ),
                   ('coder3', '47', 'shrinky'  ),
                   ('coder3', '35', 'noshrinky'),
                   ('coder3', '19', 'noshrinky'),
                   ('coder3', '3',  'noshrinky'),
                   ('coder3', '48', 'noshrinky'),
                   ('coder3', '14', 'noshrinky'),
                   ('coder3', '36', 'noshrinky'),
                   ('coder3', '36', 'shrinky'  ),
                   ('coder3', '29', 'noshrinky'),
                   ('coder3', '29', 'shrinky'  ),
                   ('coder3', '4',  'noshrinky'),
                   ('coder3', '4',  'shrinky'  ),
                   ('coder3', '1',  'shrinky'  ),
                   ('coder3', '6',  'shrinky'  ),
                   ('coder3', '41', 'shrinky'  ),
                   ('coder3', '10', 'noshrinky'),
                   ('coder3', '9',  'noshrinky'),
                   ('coder3', '38', 'shrinky'  ),
                   ('coder3', '20', 'shrinky'  ),
                   ('coder3', '11', 'shrinky'  ),
                   ('coder3', '16', 'noshrinky'),
                   ('coder3', '16', 'shrinky'  ),
                   ('coder3', '32', 'shrinky'  ),
                   ('coder3', '24', 'noshrinky'),
                   ('coder3', '2',  'noshrinky'),
                   ('coder3', '2',  'shrinky'  ),
                   ('coder3', '37', 'noshrinky'),
                   ('coder3', '34', 'noshrinky'),
                   ('coder3', '34', 'shrinky'  ),
                   ('coder3', '17', 'noshrinky'),
                   ('coder3', '17', 'shrinky'  ),
                   ('coder3', '22', 'noshrinky')]

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

for (sigma_idx, sigma) in enumerate(sigmas):
  fname = str(sigma) + '.pickle'
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
  total_confusion_matrix_HMM = np.zeros((2,2))
  total_confusion_matrix_naive = np.zeros((2,2))
  for (session_idx, (coder, subject_ID, experiment)) in enumerate(sessions_to_use):

    subject = subjects[subject_ID]
    trials_to_show = range(int(subject.experiments[experiment].datatypes['trackit'].metadata['Trial Count']))[1:]
    session_HMM_prop_correct_by_frame = []
    session_naive_prop_correct_by_frame = []
    confusion_matrix_HMM = np.zeros((2,2))
    confusion_matrix_naive = np.zeros((2,2))

    session_changepoints_HMM = []
    session_changepoints_naive = []
    session_changepoints_human = []
    for trial_idx in trials_to_show:
      trackit_trial = subject.experiments[experiment].datatypes['trackit'].trials[trial_idx]
      eyetrack_trial = subject.experiments[experiment].datatypes['eyetrack'].trials[trial_idx]

      # Extract cached HMM maximum likelihood estimate, subsampled by a factor of 6, to compare with human coding
      trial_HMM = eyetrack_trial.HMM_MLE[::6]

      # Calculate naive maximum likelihood estimate
      # naive_eyetracking.get_trackit_MLE() expects the target and distractor positions as separate arguments
      eyetrack = eyetrack_trial.data[:, 1:]
      target = trackit_trial.object_positions[trackit_trial.target_index,:,:]
      # NOTE: This works because np.delete is not in-place; it makes a copy:
      distractors = np.delete(trackit_trial.object_positions, trackit_trial.target_index, axis=0)
      trial_naive = naive_eyetracking.get_trackit_MLE(np.swapaxes(eyetrack,0,1), np.swapaxes(target,0,1), np.swapaxes(distractors,1,2))[::6]

      human_coding_filename = '../human_coded/' + coder + '/' + subject_ID + '_' + experiment + '_trial_' + str(trial_idx) + '_coding.csv'
      trial_human = []
      with open(human_coding_filename, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader, None) # Skip CSV header line
        for line in reader:
          if line[1] in ['Off Screen', 'Off Task', '']: # Ignore these frames, since they don't correspond to decodable states
            trial_human.append(-1)
          else:
            trial_human.append(int(line[1][-1:])) # NOTE: This breaks if there are >10 total objects!
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
        changepoints_HMM = get_changepoints(trial_HMM)
        changepoints_naive = get_changepoints(trial_naive)
        changepoints_human = get_changepoints(trial_human)
        confusion_matrix_HMM += generalized_2x2_confusion_matrix(changepoints_HMM, changepoints_human, max_dist=num_slack_frames)
        confusion_matrix_naive += generalized_2x2_confusion_matrix(changepoints_naive, changepoints_human, max_dist=num_slack_frames)

      total_confusion_matrix_HMM += confusion_matrix_HMM
      total_confusion_matrix_naive += confusion_matrix_naive

    # Aggregate over frames or trials within session
    HMM_prop_correct_by_session[session_idx] = np.mean(session_HMM_prop_correct_by_frame)
    naive_prop_correct_by_session[session_idx] = np.mean(session_naive_prop_correct_by_frame)
    HMM_precision_by_session[session_idx], HMM_recall_by_session[session_idx], HMM_F1_by_session[session_idx], HMM_MCC_by_session[session_idx] = classification_performance(confusion_matrix_HMM)

    naive_precision_by_session[session_idx], naive_recall_by_session[session_idx], naive_F1_by_session[session_idx], naive_MCC_by_session[session_idx] = classification_performance(confusion_matrix_naive)

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
                                             { 'prec/rec/F1' : 0.757650045421, 'MCC' : 0.750207642797 },
                                             { 'prec/rec/F1' : 0.833007711229, 'MCC' : 0.827876914201 }]
interrater_switch_performance_95CI_no_off_task = [{ 'prec/rec/F1' : 0.0541262216345, 'MCC' : 0.0542673928783 },
                                                  { 'prec/rec/F1' : 0.0466594230506, 'MCC' : 0.0471372108679 },
                                                  { 'prec/rec/F1' : 0.0406121784382, 'MCC' : 0.0411041823982 }]
interrater_average_no_off_task = 0.9556202255365588
interrater_average_95CI_no_off_task = 0.0224242939696

# Inter-rater results including frames classified as Off Task (dotted white line in plots)
interrater_switch_performance = [{ 'prec/rec/F1' : 0.219524334621, 'MCC' : 0.172656208336 },
                                 { 'prec/rec/F1' : 0.563307726573, 'MCC' : 0.536115618573 },
                                 { 'prec/rec/F1' : 0.713246989088, 'MCC' : 0.694636469785 }]
interrater_switch_performance_95CI = [{ 'prec/rec/F1' : 0.045071766976, 'MCC' : 0.0411545324437 },
                                      { 'prec/rec/F1' : 0.0540062688792, 'MCC' : 0.054302230802 },
                                      { 'prec/rec/F1' : 0.0492444774202, 'MCC' : 0.0501500003695 }]
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
if num_slack_frames < 2:
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
if num_slack_frames >= 2:
  plt.plot(sigmas, naive_rec_means, label='Naive', lw=2.5) # For high slack, Recall of Naive model is around 1, making line hard to see
else:
  plt.plot(sigmas, naive_rec_means, label='Naive')
plt.fill_between(sigmas, naive_rec_means - naive_rec_stes, naive_rec_means + naive_rec_stes, facecolor='C1', interpolate=True, alpha=0.3)
human_to_plot_upper = interrater_switch_performance_no_off_task[num_slack_frames]['prec/rec/F1']*np.array([1.0, 1.0])
human_to_plot_lower = interrater_switch_performance[num_slack_frames]['prec/rec/F1']*np.array([1.0, 1.0])
plt.plot([sigmas.min(), sigmas.max()], human_to_plot_upper, ls='--', c='k')
plt.plot([sigmas.min(), sigmas.max()], human_to_plot_lower, ls='--', c='w')
plt.fill_between([sigmas.min(), sigmas.max()], human_to_plot_lower - interrater_switch_performance_95CI[num_slack_frames]['prec/rec/F1'], human_to_plot_upper + interrater_switch_performance_95CI_no_off_task[num_slack_frames]['prec/rec/F1'], facecolor='k', interpolate=True, alpha=0.3, label='Human (Inter-rater)')
plt.ylabel('Recall', fontsize=14)
plt.ylim(bottom = 0.0, top = 1.0)
if num_slack_frames >= 2:
  plt.legend(loc='best', fancybox=True, framealpha=0.5)

plt.show()
