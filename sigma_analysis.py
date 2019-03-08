import matplotlib.pyplot as plt
import pickle
import numpy as np
import csv, math

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
HMM_means = np.zeros((len(sigmas),))
naive_means = np.zeros((len(sigmas),))
HMM_stes = np.zeros((len(sigmas),))
naive_stes = np.zeros((len(sigmas),))
for (sigma_idx, sigma) in enumerate(sigmas):
  fname = 'cache/old/cache_sigma' + str(sigma)
  print('Loading file ' + fname + '...')
  with open(fname, 'r') as f:
    subjects = pickle.load(f)

  HMM_prop_correct_by_subject = np.zeros((len(sessions_to_use),))
  naive_prop_correct_by_subject = np.zeros((len(sessions_to_use),))
  for (subject_idx, (coder, subject_ID, experiment)) in enumerate(sessions_to_use):

    subject = subjects[subject_ID]
    trials_to_show = range(int(subject.experiments[experiment].datatypes['trackit'].metadata['Trial Count']))[1:]
    subject_HMM_prop_correct_by_frame = []
    subject_naive_prop_correct_by_frame = []
    for trial_idx in trials_to_show:
      trackit_trial = subject.experiments[experiment].datatypes['trackit'].trials[trial_idx]
      eyetrack_trial = subject.experiments[experiment].datatypes['eyetrack'].trials[trial_idx]
      
      # Extract cached HMM maximum likelihood estimate, subsampled by a factor of 6
      HMM_MLE = eyetrack_trial.HMM_MLE[::6]

      # Calculate naive maximum likelihood estimate
      # naive_eyetracking.get_trackit_MLE() expects the target and distractor positions as separate arguments
      eyetrack = eyetrack_trial.data[:, 1:]
      target = trackit_trial.object_positions[trackit_trial.target_index,:,:]
      # NOTE: This works because np.delete is not in-place; it makes a copy:
      distractors = np.delete(trackit_trial.object_positions, trackit_trial.target_index, axis=0)
      naive_MLE = naive_eyetracking.get_trackit_MLE(np.swapaxes(eyetrack,0,1), np.swapaxes(target,0,1), np.swapaxes(distractors,1,2))[::6]

      # Code for just looking at time on target
      # HMM = np.array(util.performance_according_to_HMM(trackit_trial, eyetrack_trial)[::6]) # subsample (10 per second, so every 6 frames)
      # naive = np.array(util.performance_according_to_naive(trackit_trial, eyetrack_trial)[::6]) # subsample (10 per second, so every 6 frames)

      human_coding_filename = data_root + coder + '/' + subject_ID + '_' + experiment + '_trial_' + str(trial_idx) + '_coding.csv'
      human = []
      with open(human_coding_filename, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader, None) # Skip CSV header line
        for line in reader:
          if line[1] in ['Off Screen', 'Off Task', '']: # Ignore these frames, since they don't correspond to decodable states
            human.append(-1)
          else:
            human.append(int(line[1][-1:])) # NOTE: This breaks if there are >10 total objects!
          # human.append(line[1] == 'Obect 0')
      human = np.array(human)

      # Truncate to the shorter of the HMM and human coding lengths
      if HMM_MLE.shape[0] < human.shape[0]:
        human = human[:HMM_MLE.shape[0]]
      elif HMM_MLE.shape[0] > human.shape[0]:
        HMM_MLE = HMM_MLE[:human.shape[0]]
        naive_MLE = naive_MLE[:human.shape[0]]

      # Remove frames where human coding is `Off Screen'
      HMM_MLE = HMM_MLE[human >= 0]
      naive_MLE = naive_MLE[human >= 0]
      human = human[human >= 0]

      # Concatenate accuracies for current trial
      subject_HMM_prop_correct_by_frame.extend(HMM_MLE == human)
      subject_naive_prop_correct_by_frame.extend(naive_MLE == human)

    # Aggregate over frames for subject
    HMM_prop_correct_by_subject[subject_idx] = np.mean(subject_HMM_prop_correct_by_frame)
    naive_prop_correct_by_subject[subject_idx] = np.mean(subject_naive_prop_correct_by_frame)

  print('HMM accuracies:', HMM_prop_correct_by_subject[subject_idx])
  print('Naive accuracies:', naive_prop_correct_by_subject[subject_idx])

  # Aggregate over subjects
  HMM_means[sigma_idx] = np.nanmean(HMM_prop_correct_by_subject)
  naive_means[sigma_idx] = np.nanmean(naive_prop_correct_by_subject)
  HMM_stes[sigma_idx] = 1.00*np.std(HMM_prop_correct_by_subject)/math.sqrt(len(sessions_to_use))
  naive_stes[sigma_idx] = 1.00*np.std(naive_prop_correct_by_subject)/math.sqrt(len(sessions_to_use))

print(HMM_means)
print(HMM_stes)
print(naive_means)
print(naive_stes)

plt.figure()
plt.errorbar(sigmas, HMM_means, HMM_stes, label='HMM')
plt.errorbar(sigmas, naive_means, naive_stes, label='Naive')
plt.ylabel('Accuracy', fontsize=14)
plt.xlabel(r'$\sigma$' + ' parameter (pixels)', fontsize=14)
plt.ylim(bottom = 0.5, top = 1.0)
plt.legend(fontsize=14)
plt.show()
    
  # plt.figure(1)
  # plt.subplot(1, len(sigmas), sigma_idx+1)
  # util.reduce_to_means_over_trial_time(good_subjects,
  #                                      [('HMM', util.performance_according_to_HMM), ('Naive', util.performance_according_to_naive)],
  #                                      ylabel='')
  # if sigma_idx == 0:
  #   plt.ylabel('Mean Proportion of Present Trials on Target')
  # else:
  #   ax = plt.gca()
  #   ax.get_legend().remove()
  # plt.xlabel('sigma: ' + str(sigma))
  # plt.plot([600, 600], [0, 1], linestyle = '--', color = 'k')
  # plt.ylim([0, 1])
  # plt.xlim([0,900])
  # plt.figure(2)
  # plt.subplot(1, len(sigmas), sigma_idx+1)
  # plt.title('HMM and Naive Performance over Trial Number, Shrinky and Noshrinky \n(Missing Data removed)')
  # util.reduce_to_means_over_trial_num(good_subjects,
  #                                     [('HMM', util.performance_according_to_HMM), ('Naive', util.performance_according_to_naive)],
  #                                     ylabel='Mean Proportion of Present Frames on Target')
  # plt.ylim([0, 1])
# plt.show()
