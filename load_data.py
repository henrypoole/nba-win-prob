import os
import sys
import time
import math

import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 500)

# Get train/test set sizes (by game) from user input

num_internal_training_games = 4440
num_internal_testing_games = 500
num_external_testing_games = 500

#print("# Internal Training Games: ", num_internal_training_games)
#print("# Internal Testing Games: ", num_internal_testing_games)
#print("# External Testing Games: ", num_external_testing_games)
      

# LOAD DATA

start = time.time()
seasons_df_list = []
# Loop through season starting years 2002-2016
for season_start_year in range(2002, 2017):
    season_end_year = season_start_year + 1
    file_name = "{}-{}.csv".format(season_start_year, season_end_year)
    file_path = os.path.join(
        os.path.abspath("./data"),
        file_name
    )
    season_df = pd.read_csv(file_path)
    # Add season number to df, season number equal to season start year
    season_df['SEASON_START_YEAR'] = season_start_year
    # Add unique game ID string
    season_df['GAME_ID'] = season_df.GAME_NUM.apply(lambda x: "{}_{}".format(season_start_year, x))
    seasons_df_list.append(season_df)
    
seasons_df = pd.concat(seasons_df_list)
end = time.time()
print("Loaded data in {} minutes".format((end - start) / 60.))

# ADD FINAL SCORE DIFFERENCE DATA

# Group by game
grouped_by_game = seasons_df.groupby('GAME_ID')

# Get final events from each game
final_events = grouped_by_game.tail(1)

# Get final scores
final_scores = final_events[['GAME_ID', 'HT_SCORE_DIFF']]
final_scores.columns = ['GAME_ID', 'OUTCOME_HT_SCORE_DIFF']

# Left join seasons_df and final_scores on GAME_ID
seasons_df = seasons_df.merge(final_scores, on = "GAME_ID", how = "left")

# SPLIT DATAFRAME

# Create flags for seasons in event data
internal_seasons = np.arange(2002, 2014)
external_seasons = np.arange(2014, 2017)
seasons_array = seasons_df.SEASON_START_YEAR.values
internal_flag = np.isin(seasons_array, internal_seasons)
external_flag = np.isin(seasons_array, external_seasons)

# Split into internal and external groups
internal_seasons_df = seasons_df.loc[internal_flag, ]
external_seasons_df = seasons_df.loc[external_flag, ]

# CREATE TRAIN/TEST/EVAL SETS IN INTERNAL SEASONS

# Create random state and sample without replacement
internal_games = internal_seasons_df.GAME_ID.unique()
random_state = np.random.RandomState(seed = 42)
internal_games_sample = random_state.choice(internal_games, size = num_internal_training_games + num_internal_testing_games, replace = False)

# Define splits on random sample
internal_games_train = internal_games_sample[:num_internal_training_games]
internal_games_test = internal_games_sample[num_internal_training_games:]

# Check to see splits are distinct
np.intersect1d(internal_games_train, internal_games_test)

# Create flags for splits by game in event data 
internal_games_array = internal_seasons_df.GAME_ID.values
internal_train_flag = np.isin(internal_games_array, internal_games_train)
internal_test_flag = np.isin(internal_games_array, internal_games_test)

# Select model feature/label/outcome columns

pbp_features = [
    'GAME_TIME',
    'HT_POSS',
    'HT_SCORE_DIFF'
]

box_score_features = [
    'HT_SCORE',
    'HT_FOUL',
    'HT_ORBD',
    'HT_DRBD',
    'HT_TRBD',
    'HT_AST',
    'HT_STL',
    'HT_BLK',
    'HT_TOV',
    'AT_SCORE',
    'AT_FOUL',
    'AT_ORBD',
    'AT_DRBD',
    'AT_TRBD',
    'AT_AST',
    'AT_STL',
    'AT_BLK',
    'AT_TOV'
]

# Exclude player IDs
lineup_features = [
    'HT_1_S',
    'HT_1_A',
    'HT_1_GP',
    'HT_1_GS',
    'HT_1_MP',
    'HT_1_PM',
    'HT_1_MPG',
    'HT_1_PMPG',
    'HT_1_FPG',
    'HT_1_O',
    'HT_2_S',
    'HT_2_A',
    'HT_2_GP',
    'HT_2_GS',
    'HT_2_MP',
    'HT_2_PM',
    'HT_2_MPG',
    'HT_2_PMPG',
    'HT_2_FPG',
    'HT_2_O',
    'HT_3_S',
    'HT_3_A',
    'HT_3_GP',
    'HT_3_GS',
    'HT_3_MP',
    'HT_3_PM',
    'HT_3_MPG',
    'HT_3_PMPG',
    'HT_3_FPG',
    'HT_3_O',
    'HT_4_S',
    'HT_4_A',
    'HT_4_GP',
    'HT_4_GS',
    'HT_4_MP',
    'HT_4_PM',
    'HT_4_MPG',
    'HT_4_PMPG',
    'HT_4_FPG',
    'HT_4_O',
    'HT_5_S',
    'HT_5_A',
    'HT_5_GP',
    'HT_5_GS',
    'HT_5_MP',
    'HT_5_PM',
    'HT_5_MPG',
    'HT_5_PMPG',
    'HT_5_FPG',
    'HT_5_O',
    'HT_6_S',
    'HT_6_A',
    'HT_6_GP',
    'HT_6_GS',
    'HT_6_MP',
    'HT_6_PM',
    'HT_6_MPG',
    'HT_6_PMPG',
    'HT_6_FPG',
    'HT_6_O',
    'HT_7_S',
    'HT_7_A',
    'HT_7_GP',
    'HT_7_GS',
    'HT_7_MP',
    'HT_7_PM',
    'HT_7_MPG',
    'HT_7_PMPG',
    'HT_7_FPG',
    'HT_7_O',
    'HT_8_S',
    'HT_8_A',
    'HT_8_GP',
    'HT_8_GS',
    'HT_8_MP',
    'HT_8_PM',
    'HT_8_MPG',
    'HT_8_PMPG',
    'HT_8_FPG',
    'HT_8_O',
    'HT_9_S',
    'HT_9_A',
    'HT_9_GP',
    'HT_9_GS',
    'HT_9_MP',
    'HT_9_PM',
    'HT_9_MPG',
    'HT_9_PMPG',
    'HT_9_FPG',
    'HT_9_O',
    'HT_10_S',
    'HT_10_A',
    'HT_10_GP',
    'HT_10_GS',
    'HT_10_MP',
    'HT_10_PM',
    'HT_10_MPG',
    'HT_10_PMPG',
    'HT_10_FPG',
    'HT_10_O',
    'HT_11_S',
    'HT_11_A',
    'HT_11_GP',
    'HT_11_GS',
    'HT_11_MP',
    'HT_11_PM',
    'HT_11_MPG',
    'HT_11_PMPG',
    'HT_11_FPG',
    'HT_11_O',
    'HT_12_S',
    'HT_12_A',
    'HT_12_GP',
    'HT_12_GS',
    'HT_12_MP',
    'HT_12_PM',
    'HT_12_MPG',
    'HT_12_PMPG',
    'HT_12_FPG',
    'HT_12_O',
    'HT_13_S',
    'HT_13_A',
    'HT_13_GP',
    'HT_13_GS',
    'HT_13_MP',
    'HT_13_PM',
    'HT_13_MPG',
    'HT_13_PMPG',
    'HT_13_FPG',
    'HT_13_O',
    'HT_14_S',
    'HT_14_A',
    'HT_14_GP',
    'HT_14_GS',
    'HT_14_MP',
    'HT_14_PM',
    'HT_14_MPG',
    'HT_14_PMPG',
    'HT_14_FPG',
    'HT_14_O',
    'HT_15_S',
    'HT_15_A',
    'HT_15_GP',
    'HT_15_GS',
    'HT_15_MP',
    'HT_15_PM',
    'HT_15_MPG',
    'HT_15_PMPG',
    'HT_15_FPG',
    'HT_15_O',
    'AT_1_S',
    'AT_1_A',
    'AT_1_GP',
    'AT_1_GS',
    'AT_1_MP',
    'AT_1_PM',
    'AT_1_MPG',
    'AT_1_PMPG',
    'AT_1_FPG',
    'AT_1_O',
    'AT_2_S',
    'AT_2_A',
    'AT_2_GP',
    'AT_2_GS',
    'AT_2_MP',
    'AT_2_PM',
    'AT_2_MPG',
    'AT_2_PMPG',
    'AT_2_FPG',
    'AT_2_O',
    'AT_3_S',
    'AT_3_A',
    'AT_3_GP',
    'AT_3_GS',
    'AT_3_MP',
    'AT_3_PM',
    'AT_3_MPG',
    'AT_3_PMPG',
    'AT_3_FPG',
    'AT_3_O',
    'AT_4_S',
    'AT_4_A',
    'AT_4_GP',
    'AT_4_GS',
    'AT_4_MP',
    'AT_4_PM',
    'AT_4_MPG',
    'AT_4_PMPG',
    'AT_4_FPG',
    'AT_4_O',
    'AT_5_S',
    'AT_5_A',
    'AT_5_GP',
    'AT_5_GS',
    'AT_5_MP',
    'AT_5_PM',
    'AT_5_MPG',
    'AT_5_PMPG',
    'AT_5_FPG',
    'AT_5_O',
    'AT_6_S',
    'AT_6_A',
    'AT_6_GP',
    'AT_6_GS',
    'AT_6_MP',
    'AT_6_PM',
    'AT_6_MPG',
    'AT_6_PMPG',
    'AT_6_FPG',
    'AT_6_O',
    'AT_7_S',
    'AT_7_A',
    'AT_7_GP',
    'AT_7_GS',
    'AT_7_MP',
    'AT_7_PM',
    'AT_7_MPG',
    'AT_7_PMPG',
    'AT_7_FPG',
    'AT_7_O',
    'AT_8_S',
    'AT_8_A',
    'AT_8_GP',
    'AT_8_GS',
    'AT_8_MP',
    'AT_8_PM',
    'AT_8_MPG',
    'AT_8_PMPG',
    'AT_8_FPG',
    'AT_8_O',
    'AT_9_S',
    'AT_9_A',
    'AT_9_GP',
    'AT_9_GS',
    'AT_9_MP',
    'AT_9_PM',
    'AT_9_MPG',
    'AT_9_PMPG',
    'AT_9_FPG',
    'AT_9_O',
    'AT_10_S',
    'AT_10_A',
    'AT_10_GP',
    'AT_10_GS',
    'AT_10_MP',
    'AT_10_PM',
    'AT_10_MPG',
    'AT_10_PMPG',
    'AT_10_FPG',
    'AT_10_O',
    'AT_11_S',
    'AT_11_A',
    'AT_11_GP',
    'AT_11_GS',
    'AT_11_MP',
    'AT_11_PM',
    'AT_11_MPG',
    'AT_11_PMPG',
    'AT_11_FPG',
    'AT_11_O',
    'AT_12_S',
    'AT_12_A',
    'AT_12_GP',
    'AT_12_GS',
    'AT_12_MP',
    'AT_12_PM',
    'AT_12_MPG',
    'AT_12_PMPG',
    'AT_12_FPG',
    'AT_12_O',
    'AT_13_S',
    'AT_13_A',
    'AT_13_GP',
    'AT_13_GS',
    'AT_13_MP',
    'AT_13_PM',
    'AT_13_MPG',
    'AT_13_PMPG',
    'AT_13_FPG',
    'AT_13_O',
    'AT_14_S',
    'AT_14_A',
    'AT_14_GP',
    'AT_14_GS',
    'AT_14_MP',
    'AT_14_PM',
    'AT_14_MPG',
    'AT_14_PMPG',
    'AT_14_FPG',
    'AT_14_O',
    'AT_15_S',
    'AT_15_A',
    'AT_15_GP',
    'AT_15_GS',
    'AT_15_MP',
    'AT_15_PM',
    'AT_15_MPG',
    'AT_15_PMPG',
    'AT_15_FPG',
    'AT_15_O'
]

continuous_outcome = ['OUTCOME_HT_SCORE_DIFF']


# Subset internal seasons df by train/test 
# Feature sets: pbp + lineup + box score
X_train = internal_seasons_df.loc[internal_train_flag, pbp_features + lineup_features + box_score_features].to_numpy(dtype = np.float32)
X_test_internal = internal_seasons_df.loc[internal_test_flag, pbp_features + lineup_features + box_score_features].to_numpy(dtype = np.float32)
Y_train = internal_seasons_df.loc[internal_train_flag, continuous_outcome].to_numpy(dtype = np.float32)
Y_test_internal = internal_seasons_df.loc[internal_test_flag, continuous_outcome].to_numpy(dtype = np.float32)

# CREATE TEST SET FROM EXTERNAL SEASONS

X_test_external = external_seasons_df.loc[:, pbp_features + lineup_features + box_score_features].to_numpy(dtype = np.float32)
Y_test_external = external_seasons_df.loc[:, continuous_outcome].to_numpy(dtype = np.float32)

# SAVE ARRAYS AS NUMPY ARCHIVE
np.savez(
    os.path.join(os.path.abspath("./data"), "processed_data.npz"),
    X_train = X_train,
    X_test_internal = X_test_internal,
    Y_train = Y_train,
    Y_test_internal = Y_test_internal,
    X_test_external = X_test_external,
    Y_test_external = Y_test_external
)