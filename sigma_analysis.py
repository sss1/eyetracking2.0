import matplotlib.pyplot as plt
import pickle
import numpy as np
import csv, math
from sklearn.metrics import confusion_matrix

import eyetracking2_util as util
import naive_eyetracking

# BEGIN CODE FOR COMPARING HMM AND NAIVE TO HUMAN CODING

data_root = 'data/shrinky++shrinky/human_coded/'
sessions_to_use = [('coder1', 'A298','noshrinky'),
                   ('coder1', 'A326','noshrinky'),
                   ('coder1', 'A338','noshrinky'),
                   ('coder1', 'A338','shrinky'),
                   ('coder1', 'A340','noshrinky'),
                   ('coder1', 'A345','shrinky'),
                   ('coder1', 'B37', 'noshrinky'),
                   ('coder1', 'E22', 'shrinky'),
                   ('coder1', 'E52', 'noshrinky'),
                   ('coder1', 'R34', 'noshrinky'),
                   ('coder1', 'R34', 'shrinky'),
                   ('coder1', 'R37', 'noshrinky'),
                   ('coder2', 'AF7', 'noshrinky'),
                   ('coder2', 'AF7', 'shrinky'),
                   # ('coder2', 'E22', 'shrinky'),
                   # ('coder2', 'R34', 'noshrinky'),
                   # ('coder2', 'R34', 'shrinky'),
                   # ('coder2', 'R37', 'noshrinky'),
                   ('coder2', 'R37', 'shrinky'),
                   ('coder3', 'A326', 'shrinky'),
                   ('coder3', 'A345', 'noshrinky'),
                   ('coder3', 'AF8',  'shrinky'),
                   ('coder3', 'B38',  'noshrinky'),
                   # ('coder3', 'E22',  'shrinky'),
                   # ('coder3', 'R34',  'noshrinky'),
                   # ('coder3', 'R34',  'shrinky'),
                   # ('coder3', 'R37',  'noshrinky'),
                   ('coder3', 'R17',  'shrinky')]

sigmas = range(25,400,25) + range(400,1100,100)
HMM_acc_means = np.zeros((len(sigmas),))
naive_acc_means = np.zeros((len(sigmas),))
HMM_acc_stes = np.zeros((len(sigmas),))
naive_acc_stes = np.zeros((len(sigmas),))
for (sigma_idx, sigma) in enumerate(sigmas):
  fname = 'cache/old/cache_sigma' + str(sigma)
  print('Loading file ' + fname + '...')
  with open(fname, 'r') as f:
    subjects = pickle.load(f)

  HMM_prop_correct_by_session = np.zeros((len(sessions_to_use),))
  naive_prop_correct_by_session = np.zeros((len(sessions_to_use),))
  confusion_matrix_HMM = np.zeros((2,2))
  confusion_matrix_naive = np.zeros((2,2))
  for (session_idx, (coder, subject_ID, experiment)) in enumerate(sessions_to_use):

    subject = subjects[subject_ID]
    trials_to_show = range(int(subject.experiments[experiment].datatypes['trackit'].metadata['Trial Count']))[1:]
    session_HMM_prop_correct_by_frame = []
    session_naive_prop_correct_by_frame = []

    session_changepoints_HMM = []
    session_changepoints_naive = []
    session_changepoints_human = []
    for trial_idx in trials_to_show:
      trackit_trial = subject.experiments[experiment].datatypes['trackit'].trials[trial_idx]
      eyetrack_trial = subject.experiments[experiment].datatypes['eyetrack'].trials[trial_idx]
      
      # Extract cached HMM maximum likelihood estimate, subsampled by a factor of 6
      trial_HMM = eyetrack_trial.HMM_MLE[::6]

      # Calculate naive maximum likelihood estimate
      # naive_eyetracking.get_trackit_MLE() expects the target and distractor positions as separate arguments
      eyetrack = eyetrack_trial.data[:, 1:]
      target = trackit_trial.object_positions[trackit_trial.target_index,:,:]
      # NOTE: This works because np.delete is not in-place; it makes a copy:
      distractors = np.delete(trackit_trial.object_positions, trackit_trial.target_index, axis=0)
      trial_naive = naive_eyetracking.get_trackit_MLE(np.swapaxes(eyetrack,0,1), np.swapaxes(target,0,1), np.swapaxes(distractors,1,2))[::6]

      # Code for just looking at time on target
      # HMM = np.array(util.performance_according_to_HMM(trackit_trial, eyetrack_trial)[::6]) # subsample (10 per second, so every 6 frames)
      # naive = np.array(util.performance_according_to_naive(trackit_trial, eyetrack_trial)[::6]) # subsample (10 per second, so every 6 frames)

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
        confusion_matrix_HMM += confusion_matrix(changepoints_HMM, changepoints_human, labels=[False, True])
        confusion_matrix_naive += confusion_matrix(changepoints_naive, changepoints_human, labels=[False, True])

    # Aggregate over frames for session
    HMM_prop_correct_by_session[session_idx] = np.mean(session_HMM_prop_correct_by_frame)
    naive_prop_correct_by_session[session_idx] = np.mean(session_naive_prop_correct_by_frame)
  precision = float(confusion_matrix_HMM[1,1])/(confusion_matrix_HMM[1,1] + confusion_matrix_HMM[1,0])
  recall = float(confusion_matrix_HMM[1,1])/(confusion_matrix_HMM[1,1] + confusion_matrix_HMM[0,1])
  F1 = 2.0 * precision * recall / (precision + recall)
  print('HMM:', 'Precision:', precision, 'Recall:', recall, 'F1', F1)
  precision = float(confusion_matrix_naive[1,1])/(confusion_matrix_naive[1,1] + confusion_matrix_naive[1,0])
  recall = float(confusion_matrix_naive[1,1])/(confusion_matrix_naive[1,1] + confusion_matrix_naive[0,1])
  F1 = 2.0 * precision * recall / (precision + recall)
  print('naive:', 'Precision:', precision, 'Recall:', recall, 'F1', F1)

  print('HMM accuracies:', HMM_prop_correct_by_session[session_idx])
  print('Naive accuracies:', naive_prop_correct_by_session[session_idx])

  # Aggregate over sessions
  HMM_acc_means[sigma_idx] = np.nanmean(HMM_prop_correct_by_session)
  naive_acc_means[sigma_idx] = np.nanmean(naive_prop_correct_by_session)
  HMM_acc_stes[sigma_idx] = 1.00*np.std(HMM_prop_correct_by_session)/math.sqrt(len(sessions_to_use))
  naive_acc_stes[sigma_idx] = 1.00*np.std(naive_prop_correct_by_session)/math.sqrt(len(sessions_to_use))

print(HMM_acc_means)
print(HMM_acc_stes)
print(naive_acc_means)
print(naive_acc_stes)

plt.figure()
plt.errorbar(sigmas, HMM_acc_means, HMM_stes, label='HMM')
plt.errorbar(sigmas, naive_acc_means, naive_stes, label='Naive')
plt.ylabel('Accuracy', fontsize=14)
plt.xlabel(r'$\sigma$' + ' parameter (pixels)', fontsize=14)
plt.ylim(bottom = 0.5, top = 1.0)
plt.legend(fontsize=14)
plt.show()
