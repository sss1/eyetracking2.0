import numpy as np
import csv

data_root = 'data/shrinky++shrinky/human_coded/'

# BEGIN CODE FOR ASSESSING INTER-RATER RELIABILITY
coders = ['coder1', 'coder2', 'coder3']
sessions_to_use = [('E22', 'shrinky'),
                   ('R34', 'noshrinky'),
                   ('R34', 'shrinky'),
                   ('R37', 'noshrinky')
                  ]
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
        next(reader, None) # Skip CSV header line
        for line in reader:
          if line[1] == 'Obect 0': # To account for a typo that appears in many of the coding files
            frames[coder].append('Object 0')
          else:
            frames[coder].append(line[1])
          coder_file_len += 1
      # print(coder, subject_ID, experiment, trial_idx, coder_file_len)

num_frames = len(frames[coders[0]])
assert(all([len(coder_frames) == num_frames for coder_frames in frames.values()])) # Check that all coders have the same number of frames
for coder in coders:
  frames[coder] = np.array(frames[coder])

print('Number of Frames used to measure Inter-rater reliability:', num_frames)
print('Proportion of agreement:', np.mean(np.logical_and(frames['coder1'] == frames['coder3'], frames['coder1'] == frames['coder2'])))
