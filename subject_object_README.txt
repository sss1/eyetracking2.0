>>> dir(subjects['A328'])
['ID', '__doc__', '__init__', '__module__', 'experiments', 'toJSON']

>>> dir(subjects['A328'].experiments['shrinky'])
['ID', '__doc__', '__init__', '__module__', 'datatypes', 'has_all_experiment_data', 'toJSON']

>>> dir(subjects['A328'].experiments['shrinky'].datatypes['trackit'])
['__doc__', '__init__', '__module__', 'metadata', 'path', 'toJSON', 'trials']

>>> dir(subjects['A328'].experiments['shrinky'].datatypes['eyetrack'])
['__doc__', '__init__', '__module__', 'both_eyes_frames', 'interpolation', 'left_only_frames', 'missing_frames', 'path', 'proportion_imputed_frames', 'proportion_total_missing_frames', 'raw_data', 'right_only_frames', 'toJSON', 'total_frames', 'trials']

>>> dir(subjects['A328'].experiments['shrinky'].datatypes['eyetrack'].trials[0])
['HMM_MLE', '__doc__', '__init__', '__module__', 'data', 'proportion_missing', 'toJSON']

>>> dir(subjects['A328'].experiments['shrinky'].datatypes['trackit'].trials[0])
['__doc__', '__init__', '__module__', 'blinking_object_IDs', 'is_supervised', 'meta_data', 'num_objects', 'object_positions', 'rel_timestamps', 'target_index', 'timestamps', 'toJSON', 'trial_metadata']
