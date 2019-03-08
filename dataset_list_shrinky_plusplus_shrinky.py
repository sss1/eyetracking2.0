# root = "/home/sss1/Desktop/projects/eyetracking/data/" # Office desktop
root = "/home/painkiller/Desktop/academic/projects/eyetracking2.0/data/" # Laptop
# root = "/home/sss1/Desktop/academic/projects/eyetracking/data/" # Home desktop

dataset_list = { \
  'noshrinky_eyetrack' : \
    [('A272', 'A272_noshrinky_10_2_2018_10_15.csv'), \
     ('A276', 'a276noshrinky_9_25_2018_10_24.csv'), \
     ('A283', 'A283_noshrinky_10_2_2018_10_42.csv'), \
     ('A290', 'A290_noshrinky_9_26_2018_10_36.csv'), \
     ('A294', 'a294noshrinky_9_25_2018_13_26.csv'), \
     ('A295', 'a295noshrinky_9_27_2018_13_52.csv'), \
     ('A297', 'a297noshrinky_9_27_2018_13_38.csv'), \
     ('A298', 'a298noshrinky_9_25_2018_13_38.csv'), \
     ('A326', 'a326noshrinky_10_3_2018_10_25.csv'), \
     ('A327', 'A327_noshrinky_10_2_2018_9_59.csv'), \
     ('A328', 'a328noshrinky_9_27_2018_10_13.csv'), \
     ('A338', 'a338noshrinky_9_27_2018_10_28.csv'), \
     ('A339', 'A339_noshrinky_10_1_2018_13_44.csv'), \
     ('A340', 'A340_noshrinky_9_26_2018_13_49.csv'), \
     ('A343', 'a343noshrinky_9_20_2018_8_53.csv'), \
     ('A344', 'A344_noshrinky_9_19_2018_9_52.csv'), \
     ('A345', 'a345noshrinky_9_25_2018_9_54.csv'), \
     ('A347', 'a347noshrinky_9_27_2018_8_42.csv'), \
     ('A351', 'a351noshrinky_9_25_2018_12_2.csv'), \
     ('A352', 'A352_noshrinky_9_19_2018_10_12.csv'), \
     ('A353', 'a353noshrinky_9_27_2018_9_2.csv'), \
     ('A380', 'A380_noshrinky_12_11_2018_13_7.csv'), \
     ('AF10', 'AF10noshrinky_5_7_2018_10_57.csv'), \
     ('AF11', 'AF11noshrinky_5_7_2018_10_17.csv'), \
     ('AF2',  'AF2noshrinky_5_7_2018_10_30.csv'), \
     ('AF7',  'AF7noshrinky_5_14_2018_9_36.csv'), \
     ('AF8',  'AF8noshrinky_5_14_2018_9_21.csv'), \
     ('B31',  'B31noshrinky_5_9_2018_10_40.csv'), \
     ('B33',  'B33noshrinky_5_2_2018_9_45.csv'), \
     ('B37',  'B37noshrinky_5_9_2018_10_26.csv'), \
     ('B38',  'B38noshrinky_5_2_2018_11_16.csv'), \
     ('B40',  'B40noshrinky_5_23_2018_13_43.csv'), \
     ('B42',  'B42noshrinky_5_23_2018_13_58.csv'), \
     ('B56',  'B56_noshrinky_12_11_2018_9_10.csv'), \
     ('B61',  'B61_noshrinky_12_13_2018_9_39.csv'), \
     ('B63',  'B63_noshrinky_12_11_2018_9_27.csv'), \
     ('C59',  'C59_noshrinky_12_18_2018_9_45.csv'), \
     ('D5',   'D5noshrinky_5_3_2018_9_24.csv'), \
     ('D6',   'D6noshrinky_5_1_2018_10_11.csv'), \
     ('E22',  'E22noshrinky_4_27_2018_9_41.csv'), \
     ('E29',  'E29noshrinky_4_27_2018_10_47.csv'), \
     ('E3',   'E3noshrinky_4_30_2018_10_13.csv'), \
     ('E51',  'E51noshrinky_4_30_2018_10_38.csv'), \
     ('E52',  'E52noshrinky_4_27_2018_10_39.csv'), \
     ('L177', 'L177_noshrinky_12_13_2018_10_55.csv'), \
     ('M114', 'M114noshrinky_5_11_2018_13_15.csv'), \
     ('M115', 'M115noshrinky_5_11_2018_13_27.csv'), \
     ('M117', 'M117noshrinky_5_4_2018_12_51.csv'), \
     ('M137', 'M137noshrinky_5_11_2018_10_33.csv'), \
     ('M138', 'M138noshrinky_5_11_2018_10_53.csv'), \
     ('M139', 'M139noshrinky_5_4_2018_10_33.csv'), \
     ('M140', 'M140noshrinky_5_11_2018_10_16.csv'), \
     ('M141', 'M141noshrinky_5_4_2018_9_28.csv'), \
     ('M142', 'M142noshrinky_5_4_2018_13_48.csv'), \
     ('M143', 'M143noshrinky_5_4_2018_9_40.csv'), \
     ('M144', 'M144noshrinky_5_4_2018_11_9.csv'), \
     ('M146', 'M146noshrinky_5_4_2018_10_16.csv'), \
     ('M147', 'M147noshrinky_5_11_2018_11_10.csv'), \
     ('M191', 'M191_noshrinky_12_7_2018_10_8.csv'), \
     ('M193', 'M193_noshrinky_12_10_2018_10_47.csv'), \
     ('M197', 'M197_noshrinky_12_10_2018_10_18.csv'), \
     ('M198', 'm198_noshrinky_1_14_2019_9_39.csv'), \
     ('M199', 'm199_noshrinky_1_14_2019_9_51.csv'), \
     ('M201', 'M201_noshrinky_12_10_2018_9_52.csv'), \
     ('M204', 'M204_noshrinky_12_7_2018_10_33.csv'), \
     ('M206', 'M206_noshrinky_12_10_2018_10_5.csv'), \
     ('R12',  'R12noshrinky_4_24_2018_9_50.csv'), \
     ('R17',  'R17noshrinky_5_8_2018_9_57.csv'), \
     ('R34',  'R34noshrinky_5_8_2018_9_44.csv'), \
     ('R35',  'R35noshrinky_4_26_2018_10_27.csv'), \
     ('R37',  'R37noshrinky_4_24_2018_10_1.csv')], \
  'shrinky_eyetrack' : \
    [('A272', 'A272_shrinky_9_26_2018_14_12.csv'), \
     ('A276', 'a276shrinky_9_27_2018_13_25.csv'), \
     ('A290', 'A290_shrinky_10_1_2018_14_2.csv'), \
     ('A294', 'A294_shrinky_10_2_2018_10_27.csv'), \
     ('A295', 'a295shrinky_9_25_2018_13_51.csv'), \
     ('A297', 'a297shrinky_9_25_2018_10_44.csv'), \
     ('A298', 'a298shrinky_9_27_2018_10_40.csv'), \
     ('A326', 'A326_shrinky_9_26_2018_12_5.csv'), \
     ('A327', 'A327__shrinky_9_26_2018_12_20.csv'), \
     ('A328', 'a328shrinky_10_3_2018_9_52.csv'), \
     ('A338', 'A338shrinky_9_25_2018_10_10.csv'), \
     ('A339', 'A339_shrinky_9_26_2018_10_54.csv'), \
     ('A340', 'A340_shrinky_10_1_2018_13_30.csv'), \
     ('A343', 'a343shrinky_9_25_2018_12_17.csv'), \
     ('A344', 'A344_shrinky_9_26_2018_8_55.csv'), \
     ('A345', 'a345shrinky_9_20_2018_10_8.csv'), \
     ('A347', 'A347shrinky_9_25_2018_8_47.csv'), \
     ('A351', 'a351shrinky_9_27_2018_12_18.csv'), \
     ('A352', 'A352_shrinky_9_26_2018_9_7.csv'), \
     ('A353', 'a353shrinky_10_4_2018_9_50.csv'), \
     ('A380', 'A380_shrinky_12_13_2018_12_43.csv'), \
     ('AF10', 'AF10shrinky_5_14_2018_9_48.csv'), \
     ('AF11', 'AF11shrinky_5_14_2018_10_20.csv'), \
     ('AF2',  'AF2shrinky_5_14_2018_10_9.csv'), \
     ('AF5',  'AF5shrinky_5_7_2018_10_43.csv'), \
     ('AF7',  'AF7shrinky_5_7_2018_9_45.csv'), \
     ('AF8',  'AF8shrinky_5_7_2018_10_1.csv'), \
     ('B33',  'B33shrinky_5_9_2018_10_9.csv'), \
     ('B37',  'B37shrinky_5_2_2018_10_20.csv'), \
     ('B38',  'B38shrinky_5_9_2018_10_56.csv'), \
     ('B40',  'B40shrinky_5_2_2018_10_42.csv'), \
     ('B42',  'B42shrinky_5_2_2018_10_58.csv'), \
     ('B56',  'B56_shrinky_12_13_2018_9_7.csv'), \
     ('B61',  'B61_shrinky_12_11_2018_8_57.csv'), \
     ('B63',  'B63_shrinky_12_13_2018_9_23.csv'), \
     ('C59',  'C59_shrinky_12_11_2018_10_30.csv'), \
     ('D5',   'D5shrinky_5_1_2018_10_31.csv'), \
     ('D6',   'D6shrinky_5_3_2018_9_42.csv'), \
     ('E22',  'E22shrinky_4_30_2018_9_46.csv'), \
     ('E29',  'E29shrinky_4_30_2018_10_0.csv'), \
     ('E3',   'E3shrinky_4_27_2018_10_23.csv'), \
     ('E51',  'E51shrinky_4_27_2018_9_56.csv'), \
     ('E52',  'E52shrinky_4_30_2018_10_26.csv'), \
     ('L177', 'L177_shrinky_12_6_2018_9_18.csv'), \
     ('M114', 'M114shrinky_5_4_2018_13_15.csv'), \
     ('M115', 'M115shrinky_5_4_2018_13_35.csv'), \
     ('M117', 'M117shrinky_5_11_2018_13_3.csv'), \
     ('M138', 'M138shrinky_5_4_2018_10_46.csv'), \
     ('M139', 'M139shrinky_5_11_2018_10_43.csv'), \
     ('M140', 'M140shrinky_5_4_2018_9_51.csv'), \
     ('M141', 'M141shrinky_5_11_2018_10_4.csv'), \
     ('M143', 'M143shrinky_5_11_2018_9_55.csv'), \
     ('M144', 'M144shrinky_5_11_2018_11_2.csv'), \
     ('M146', 'M146shrinky_5_11_2018_10_25.csv'), \
     ('M147', 'M147shrinky_5_4_2018_10_59.csv'), \
     ('M191', 'M191_shrinky_12_10_2018_10_35.csv'), \
     ('M193', 'M193_shrinky_12_14_2018_10_38.csv'), \
     ('M197', 'M197_shrinky_12_7_2018_9_53.csv'), \
     ('M198', 'm198_shrinky_1_7_2019_9_38.csv'), \
     ('M201', 'M201_shrinky_12_7_2018_10_21.csv'), \
     ('M206', 'M206_shrinky_12_7_2018_10_48.csv'), \
     ('R11',  'R11shrinky_4_24_2018_9_28.csv'), \
     ('R12',  'R12shrinky_4_26_2018_9_33.csv'), \
     ('R17',  'R17shrinky_4_24_2018_10_34.csv'), \
     ('R34',  'R34shrinky_4_26_2018_10_10.csv'), \
     ('R35',  'R35shrinky_5_8_2018_10_8.csv'), \
     ('R37',  'R37shrinky_5_8_2018_9_26.csv')], \
  'noshrinky_trackit' : \
    [('A272', 'A272_noshrinky.csv'), \
     ('A276', 'a276noshrinky.csv'), \
     ('A283', 'A283_noshrinky.csv'), \
     ('A290', 'A290_noshrinky.csv'), \
     ('A294', 'a294noshrinky.csv'), \
     ('A295', 'a295noshrinky.csv'), \
     ('A297', 'a297noshrinky.csv'), \
     ('A298', 'a298noshrinky.csv'), \
     ('A326', 'a326noshrinky.csv'), \
     ('A327', 'A327_noshrinky.csv'), \
     ('A328', 'a328noshrinky.csv'), \
     ('A338', 'a338noshrinky.csv'), \
     ('A339', 'A339_noshrinky.csv'), \
     ('A340', 'A340_nosrhinky.csv'), \
     ('A343', 'A343noshrinky.csv'), \
     ('A344', 'A344_noshrinky.csv'), \
     ('A345', 'a345noshrinky.csv'), \
     ('A347', 'a347noshrinky.csv'), \
     ('A351', 'a351noshrinky.csv'), \
     ('A352', 'A352_noshrinky.csv'), \
     ('A353', 'a353noshrinky.csv'), \
     ('A380', 'A380_noshrinky.csv'), \
     ('AF10', 'AF10noshrinky.csv'), \
     ('AF11', 'AF11noshrinky.csv'), \
     ('AF2',  'AF2noshrinky.csv'), \
     ('AF7',  'AF7noshrinky.csv'), \
     ('AF8',  'AF8noshrinky.csv'), \
     ('B31',  'B31noshrinky.csv'), \
     ('B33',  'B33noshrinky.csv'), \
     ('B37',  'B37noshrinky.csv'), \
     ('B38',  'B38noshrinky.csv'), \
     ('B40',  'B40noshrinky.csv'), \
     ('B42',  'B42noshrinky.csv'), \
     ('B56',  'B56_noshrinky.csv'), \
     ('B61',  'B61_noshrinky.csv'), \
     ('B63',  'B63_noshrinky.csv'), \
     ('C59',  'C59_noshrinky.csv'), \
     ('D5',   'D5noshrinky.csv'), \
     ('D6',   'D6noshrinky.csv'), \
     ('E22',  'E22noshrinky.csv'), \
     ('E29',  'E29noshrinky.csv'), \
     ('E3',   'E3noshrinky.csv'), \
     ('E51',  'E51noshrinky.csv'), \
     ('E52',  'E52noshrinky.csv'), \
     ('L177', 'L177_noshrinky.csv'), \
     ('M114', 'M114noshrinky.csv'), \
     ('M115', 'M115noshrinky.csv'), \
     ('M117', 'M117noshrinky.csv'), \
     ('M137', 'M137noshrinky.csv'), \
     ('M138', 'M138noshrinky.csv'), \
     ('M139', 'M139noshrinky.csv'), \
     ('M140', 'M140noshrinky.csv'), \
     ('M141', 'M141noshrinky.csv'), \
     ('M142', 'M142noshrinky.csv'), \
     ('M143', 'M143noshrinky.csv'), \
     ('M144', 'M144noshrinky.csv'), \
     ('M146', 'M146noshrinky.csv'), \
     ('M147', 'M147noshrinky.csv'), \
     ('M191', 'M191_noshrinky.csv'), \
     ('M193', 'M193_noshrinky.csv'), \
     ('M197', 'M197_noshrinky.csv'), \
     ('M198', 'M198_noshrinky.csv'), \
     ('M199', 'M199_noshrinky.csv'), \
     ('M201', 'M201_noshrinky.csv'), \
     ('M204', 'M204_noshrinky.csv'), \
     ('M206', 'M206_noshrinky.csv'), \
     ('R11',  'R11noshrinky.csv'), \
     ('R12',  'R12noshrinky.csv'), \
     ('R17',  'R17noshrinky.csv'), \
     ('R34',  'R34noshrinky.csv'), \
     ('R35',  'R35noshrinky.csv'), \
     ('R37',  'R37noshrinky.csv')], \
  'shrinky_trackit' : \
    [('A272', 'A272_shrinky.csv'), \
     ('A276', 'a276shrinky.csv'), \
     ('A290', 'A290_shrinky.csv'), \
     ('A294', 'A294_shrinky.csv'), \
     ('A295', 'a295shrinky.csv'), \
     ('A297', 'a297shrinky.csv'), \
     ('A298', 'a298shrinky.csv'), \
     ('A326', 'A326_shrinky.csv'), \
     ('A327', 'A327_shrinky.csv'), \
     ('A328', 'a328shrinky.csv'), \
     ('A338', 'a338shrinky.csv'), \
     ('A339', 'A339_shrinky.csv'), \
     ('A340', 'A340_shrink.csv'), \
     ('A343', 'a343shrinky.csv'), \
     ('A344', 'A344_shrinky.csv'), \
     ('A345', 'a345shrinky.csv'), \
     ('A347', 'a347shrinky.csv'), \
     ('A351', 'a351shrinky.csv'), \
     ('A352', 'A352_shrinky.csv'), \
     ('A353', 'a353shrinky.csv'), \
     ('A380', 'A380_shrinky.csv'), \
     ('AF10', 'AF10shrinky.csv'), \
     ('AF11', 'AF11shrinky.csv'), \
     ('AF2',  'AF2shrinky.csv'), \
     ('AF5',  'AF5shrinky.csv'), \
     ('AF7',  'AF7shrinky.csv'), \
     ('AF8',  'AF8shrinky.csv'), \
     ('B31',  'B31shrinky.csv'), \
     ('B33',  'B33shrinky.csv'), \
     ('B37',  'B37shrinky.csv'), \
     ('B38',  'B38shrinky.csv'), \
     ('B40',  'B40shrinky.csv'), \
     ('B42',  'B42shrinky.csv'), \
     ('B56',  'B56_shrinky.csv'), \
     ('B61',  'B61_shrinky.csv'), \
     ('B63',  'B63_shrinky.csv'), \
     ('C59',  'C59_shrinky.csv'), \
     ('D5',   'D5shrinky.csv'), \
     ('D6',   'D6shrinky.csv'), \
     ('E22',  'E22shrinky.csv'), \
     ('E29',  'E29shrinky.csv'), \
     ('E3',   'E3shrinky.csv'), \
     ('E51',  'E51shrinky.csv'), \
     ('E52',  'E52shrinky.csv'), \
     ('L177', 'L177_shrinky.csv'), \
     ('M114', 'M114shrinky.csv'), \
     ('M115', 'M115shrinky.csv'), \
     ('M117', 'M117shrinky.csv'), \
     ('M137', 'M137shrinky.csv'), \
     ('M138', 'M138shrinky.csv'), \
     ('M139', 'M139shrinky.csv'), \
     ('M140', 'M140shrinky.csv'), \
     ('M141', 'M141shrinky.csv'), \
     ('M143', 'M143shrinky.csv'), \
     ('M144', 'M144shrinky.csv'), \
     ('M146', 'M146shrinky.csv'), \
     ('M147', 'M147shrinky.csv'), \
     ('M191', 'M191_shrinky.csv'), \
     ('M193', 'M193_shrinky.csv'), \
     ('M197', 'M197_shrinky.csv'), \
     ('M198', 'M198_shrinky.csv'), \
     ('M201', 'M201_shrinky.csv'), \
     ('M206', 'M206_shrinky.csv'), \
     ('R11',  'R11shrinky.csv'), \
     ('R12',  'R12shrinky.csv'), \
     ('R17',  'R17shrinky.csv'), \
     ('R34',  'R34shrinky.csv'), \
     ('R35',  'R35shrinky.csv'), \
     ('R37',  'R37shrinky.csv')] \
  # 'mean_age' : { \
  #   'A272' : (5.72 + 5.74)/2, \
  #   'A276' : (5.51 + 5.52)/2, \
  #   'A283' :         5.23   , \
  #   'A290' : (4.90 + 4.91)/2, \
  #   'A294' : (4.76 + 4.78)/2, \
  #   'A295' : (4.74 + 4.75)/2, \
  #   'A297' : (5.72 + 5.73)/2, \
  #   'A298' : (5.25 + 5.26)/2, \
  #   'A326' : (4.06 + 4.08)/2, \
  #   'A327' : (4.00 + 4.02)/2, \
  #   'A328' : (3.96 + 3.98)/2, \
  #   'A338' : (5.40 + 5.41)/2, \
  #   'A339' : (5.06 + 5.07)/2, \
  #   'A340' : (5.04 + 5.05)/2, \
  #   'A343' : (4.41 + 4.42)/2, \
  #   'A344' : (4.13 + 4.15)/2, \
  #   'A345' : (4.09 + 4.10)/2, \
  #   'A347' : (3.85 + 3.85)/2, \
  #   'A351' : (3.85 + 3.85)/2, \
  #   'A352' : (4.13 + 4.15)/2, \
  #   'A353' : (3.85 + 3.87)/2, \
  #   'AF10' : (3.97 + 3.95)/2, \
  #   'AF11' : (4.53 + 4.51)/2, \
  #   'AF2'  : (4.21 + 4.19)/2, \
  #   'AF5'  :  3.65          , \
  #   'AF7'  : (4.01 + 4.03)/2, \
  #   'AF8'  : (4.01 + 4.03)/2, \
  #   'B31'  :         5.34   , \
  #   'B33'  : (5.18 + 5.16)/2, \
  #   'B37'  : (4.73 + 4.75)/2, \
  #   'B38'  : (4.2  + 4.18)/2, \
  #   'B40'  : (4.16 + 4.22)/2, \
  #   'B42'  : (4.39 + 4.45)/2, \
  #   'D5'   : (3.61 + 3.61)/2, \
  #   'D6'   : (3.76 + 3.76)/2, \
  #   'E22'  : (5.67 + 5.66)/2, \
  #   'E29'  : (5.96 + 5.95)/2, \
  #   'E3'   : (5.86 + 5.87)/2, \
  #   'E51'  : (5.93 + 5.94)/2, \
  #   'E52'  : (5.97 + 5.96)/2, \
  #   'M114' : (3.06 + 3.08)/2, \
  #   'M115' : (3.78 + 3.8 )/2, \
  #   'M117' : (3.67 + 3.65)/2, \
  #   'M137' :         4.5    , \
  #   'M138' : (3.84 + 3.86)/2, \
  #   'M139' : (3.72 + 3.7 )/2, \
  #   'M140' : (4.31 + 4.33)/2, \
  #   'M141' : (4.25 + 4.23)/2, \
  #   'M142' :         4.29   , \
  #   'M143' : (4.22 + 4.21)/2, \
  #   'M144' : (3.98 + 3.96)/2, \
  #   'M146' : (4.59 + 4.57)/2, \
  #   'M147' : (3.88 + 3.9 )/2, \
  #   'R11'  :  4.09          , \
  #   'R12'  : (4.09 + 4.09)/2, \
  #   'R17'  : (4.77 + 4.81)/2, \
  #   'R34'  : (5.05 + 5.08)/2, \
  #   'R35'  : (5.39 + 5.36)/2, \
  #   'R37'  : (4.38 + 4.35)/2, \
  # }\
}

dataset_list['noshrinky_eyetrack'] = [(subject_ID, root + 'shrinky++shrinky/eyetracking/noshrinky/' + fname) for (subject_ID, fname) in dataset_list['noshrinky_eyetrack']]
dataset_list['shrinky_eyetrack'] = [(subject_ID, root + 'shrinky++shrinky/eyetracking/shrinky/' + fname) for (subject_ID, fname) in dataset_list['shrinky_eyetrack']]
dataset_list['noshrinky_trackit'] = [(subject_ID, root + 'shrinky++shrinky/trackit/noshrinky/' + fname) for (subject_ID, fname) in dataset_list['noshrinky_trackit']]
dataset_list['shrinky_trackit'] = [(subject_ID, root + 'shrinky++shrinky/trackit/shrinky/' + fname) for (subject_ID, fname) in dataset_list['shrinky_trackit']]
