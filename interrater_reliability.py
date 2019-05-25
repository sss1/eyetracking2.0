import numpy as np
import csv
import math

import eyetracking2_util as util

data_root = 'data/shrinky++shrinky/human_coded/'

num_slack_frames = 2 # Temporal slack for computing confusion matrices. Higher gives more lenient classification results.

# BEGIN CODE FOR ASSESSING INTER-RATER RELIABILITY
coders = ['coder1', 'coder3']
sessions_to_use = [('R34',  'noshrinky'),
                   ('R34',  'shrinky'),
                   ('E22',  'shrinky'),
                   ('R37',  'noshrinky'),
                   ('M206', 'noshrinky'),
                   ('M198', 'shrinky'),
                   ('A298', 'shrinky'),
                   ('A297', 'shrinky'),
                   ('AF11', 'noshrinky'),
                   ('E3',   'noshrinky'),
                   ('E51',  'shrinky'),
                   ('M146', 'shrinky'),
                   ('A294', 'noshrinky'),
                   ('A276', 'shrinky'),
                   ('AF2',  'noshrinky'),
                   ('A380', 'noshrinky'),
                   ('R35',  'shrinky'),
                   ('E29',  'noshrinky')]
num_sessions = len(sessions_to_use)
trials_to_show = range(1, 11)

# Read in judgements for each coder
frames = {coder : [] for coder in coders}
for (subject_idx, (subject_ID, experiment)) in enumerate(sessions_to_use):
  for trial_idx in trials_to_show:
    for coder in coders:
      coding_filename = data_root + coder + '/' + subject_ID + '_' + experiment + '_trial_' + str(trial_idx) + '_coding.csv'
      coder_file_len = 0
      with open(coding_filename, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for _ in range(5):
          next(reader, None) # Skip CSV header line
        for line in reader:
          if line[1] == 'Obect 0': # To account for a typo that appears in many of the coding files
            frames[coder].append('Object 0')
          else:
            frames[coder].append(line[1])
          coder_file_len += 1
      # print(coder, subject_ID, experiment, trial_idx, coder_file_len)
    # print('Subject: ' + subject_ID + ' Trial: ' + str(trial_idx) + ' Proportion of agreement:', np.mean(np.array(frames['coder1']) == np.array(frames['coder3'])))

num_frames = len(frames[coders[0]])
assert(all([len(coder_frames) == num_frames for coder_frames in frames.values()])) # Check that all coders have the same number of frames
for coder in coders:
  frames[coder] = np.array(frames[coder])

# Remove frames coded as "Off Screen" by either coder
to_remove = (np.logical_and(frames['coder1'] == 'Off Screen', frames['coder3'] == 'Off Screen'))
frames['coder3'] = frames['coder3'][frames['coder1'] != 'Off Screen']
frames['coder1'] = frames['coder1'][frames['coder1'] != 'Off Screen']
frames['coder1'] = frames['coder1'][frames['coder3'] != 'Off Screen']
frames['coder3'] = frames['coder3'][frames['coder3'] != 'Off Screen']

# Optionally remove frames coded as "Off Task" by either coder
remove_any_off_task = False
if remove_any_off_task:
  frames['coder3'] = frames['coder3'][frames['coder1'] != 'Off Task']
  frames['coder1'] = frames['coder1'][frames['coder1'] != 'Off Task']
  frames['coder1'] = frames['coder1'][frames['coder3'] != 'Off Task']
  frames['coder3'] = frames['coder3'][frames['coder3'] != 'Off Task']

print('Number of Frames used to measure Inter-rater reliability:', num_frames)
prop_agreement = np.mean(frames['coder1'] == frames['coder3'])
prop_agreement_SD = math.sqrt(prop_agreement*(1-prop_agreement))/num_sessions
print('Proportion of agreement: ' + str(prop_agreement) + ' +/- ' + str(1.96 * prop_agreement_SD))

changepoints_coder1 = util.get_changepoints(frames['coder1'])
changepoints_coder3 = util.get_changepoints(frames['coder3'])

precision, recall, _, MCC = util.classification_performance(util.generalized_2x2_confusion_matrix(changepoints_coder1, changepoints_coder3, max_dist=0))
f1 = (precision + recall)/2.0 # Since coders are symmetric, precision = recall = F1
prec_rec_f1_SD = math.sqrt(f1*(1-f1))/num_sessions
MCC_SD = math.sqrt(MCC*(1-MCC))/num_sessions
print('Averages over coders at slack 0:')
print('Precision == Recall == F1: ' + str(f1) + ' +/- ' + str(1.96 * prec_rec_f1_SD) + ',  MCC: ' + str(MCC) + ' +/- ' + str(1.96 * MCC_SD))

precision, recall, _, MCC = util.classification_performance(util.generalized_2x2_confusion_matrix(changepoints_coder1, changepoints_coder3, max_dist=1))
f1 = (precision + recall)/2.0 # Since coders are symmetric, precision = recall = F1
prec_rec_f1_SD = math.sqrt(f1*(1-f1))/num_sessions
MCC_SD = math.sqrt(MCC*(1-MCC))/num_sessions
print('Averages over coders at slack 1:')
print('Precision == Recall == F1: ' + str(f1) + ' +/- ' + str(1.96 * prec_rec_f1_SD) + ',  MCC: ' + str(MCC) + ' +/- ' + str(1.96 * MCC_SD))

precision, recall, _, MCC = util.classification_performance(util.generalized_2x2_confusion_matrix(changepoints_coder1, changepoints_coder3, max_dist=2))
f1 = (precision + recall)/2.0 # Since coders are symmetric, precision = recall = F1
prec_rec_f1_SD = math.sqrt(f1*(1-f1))/num_sessions
MCC_SD = math.sqrt(MCC*(1-MCC))/num_sessions
print('Averages over coders at slack 2:')
print('Precision == Recall == F1: ' + str(f1) + ' +/- ' + str(1.96 * prec_rec_f1_SD) + ',  MCC: ' + str(MCC) + ' +/- ' + str(1.96 * MCC_SD))

precision, recall, _, MCC = util.classification_performance(util.generalized_2x2_confusion_matrix(changepoints_coder1, changepoints_coder3, max_dist=3))
f1 = (precision + recall)/2.0 # Since coders are symmetric, precision = recall = F1
prec_rec_f1_SD = math.sqrt(f1*(1-f1))/num_sessions
MCC_SD = math.sqrt(MCC*(1-MCC))/num_sessions
print('Averages over coders at slack 3:')
print('Precision == Recall == F1: ' + str(f1) + ' +/- ' + str(1.96 * prec_rec_f1_SD) + ',  MCC: ' + str(MCC) + ' +/- ' + str(1.96 * MCC_SD))
