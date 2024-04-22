import pandas as pd
import numpy as np
import pickle
import chardet

def convert_bases_features(df):
    df.on_1b.where(df.on_1b.isna(),1, inplace=True)
    df.on_1b.fillna(0, inplace=True)
    df.on_2b.where(df.on_2b.isna(),1, inplace=True)
    df.on_2b.fillna(0, inplace=True)
    df.on_3b.where(df.on_3b.isna(),1, inplace=True)
    df.on_3b.fillna(0, inplace=True)

    df.on_1b = df.on_1b.astype(int).astype(str)
    df.on_2b = df.on_2b.astype(int).astype(str)
    df.on_3b = df.on_3b.astype(int).astype(str)
    return df

def convert_type(df):
    df.pitch_type = df.pitch_type.astype("category")
    df.inning = df.inning.astype("category")
    df.p_throws = df.p_throws.astype("category")
    df.stand = df.stand.astype("category")
    df.top = df.top.astype("boolean")
    df.prev_pitch_type = df.prev_pitch_type.astype("category")
    df.prev_type = df.stand.astype("category")
    return df

def add_prev_pitch_type_tendencies(pitcher_specific_df, K, names):
    k_previous_types_dist = np.zeros((pitcher_specific_df.shape[0], len(all_typnamese_names)))
    k_previous_types_dist[0:K, :] = np.nan
    
    for i in range(K, pitcher_specific_df.shape[0]):    # first pitch has no previous, store with nan
        min_index = max(0,i-K)
        slice = pitcher_specific_df.iloc[min_index:i,:] # get K previous pitches
        type_distribution = slice.pitch_type.value_counts(normalize=True)   # get rates each type thrown
        not_present_types = list(set(names) - set(type_distribution.index))  # get all the types not thrown by that pitcher within K pitches
        not_present_types_with_zeros = pd.Series(len(not_present_types)*[0.0], index = not_present_types)   # identify that these types were thrown 0% of the time
        full_type_distribution = pd.concat([type_distribution, not_present_types_with_zeros]).sort_index()  # sort so order the same
        k_previous_types_dist[i] = full_type_distribution   # store in array

    return pd.DataFrame(k_previous_types_dist, columns=names)

def add_pitch_strike_tendencies(pitcher_specific_df, K, names):
    k_previous_types_dist = np.zeros((pitcher_specific_df.shape[0], len(names)))
    k_previous_types_dist[0:K, :] = np.nan
    
    for i in range(K, pitcher_specific_df.shape[0]):    # first pitch has no previous, store with nan
        min_index = max(0,i-K)
        slice = pitcher_specific_df.iloc[min_index:i,:] # get K previous pitches
        type_distribution = slice.type.value_counts(normalize=True)   # get rates each type thrown
        not_present_types = list(set(names) - set(type_distribution.index))  # get all the types not thrown by that pitcher within K pitches
        not_present_types_with_zeros = pd.Series(len(not_present_types)*[0.0], index = not_present_types)   # identify that these types were thrown 0% of the time
        full_type_distribution = pd.concat([type_distribution, not_present_types_with_zeros]).sort_index()  # sort so order the same
        k_previous_types_dist[i] = full_type_distribution   # store in array

    return pd.DataFrame(k_previous_types_dist, columns=names)

# don't start counting till K_start
def add_pitcher_historical_tendencies(pitcher_specific_df, K_start, names):
    k_previous_types_dist = np.zeros((pitcher_specific_df.shape[0], len(names)))
    k_previous_types_dist[0:K_start, :] = np.nan
    
    for i in range(K_start, pitcher_specific_df.shape[0]):    # first pitch has no previous, store with nan
        slice = pitcher_specific_df.iloc[0:i,:] # get K previous pitches
        type_distribution = slice.pitch_type.value_counts(normalize=True)   # get rates each type thrown
        not_present_types = list(set(names) - set(type_distribution.index))  # get all the types not thrown by that pitcher within K pitches
        not_present_types_with_zeros = pd.Series(len(not_present_types)*[0.0], index = not_present_types)   # identify that these types were thrown 0% of the time
        full_type_distribution = pd.concat([type_distribution, not_present_types_with_zeros]).sort_index()  # sort so order the same
        k_previous_types_dist[i] = full_type_distribution   # store in array

    return pd.DataFrame(k_previous_types_dist, columns=names)

# very unoptimized, just first working implementation
def add_pit_v_bat_matchups(advanced_dataset, names):
    pit_v_bat_types_dist = np.zeros((advanced_dataset.shape[0], len(names)))
    # only take last 20 instances
    for i in range(advanced_dataset.shape[0]):
        pitcher_id = advanced_dataset.iloc[i,:]['pitcher_id']
        batter_id = advanced_dataset.iloc[i,:]['batter_id']
        slice = advanced_dataset.iloc[0:i,:].query("pitcher_id == @pitcher_id and batter_id == @batter_id").tail(20)
        if slice.shape[0] == 0:
            pit_v_bat_types_dist[i] = np.nan    
        else:
            type_distribution = slice.pitch_type.value_counts(normalize=True)   # get rates each type thrown
            not_present_types = list(set(names) - set(type_distribution.index))  # get all the types not thrown by that pitcher within K pitches
            not_present_types_with_zeros = pd.Series(len(not_present_types)*[0.0], index = not_present_types)   # identify that these types were thrown 0% of the time
            full_type_distribution = pd.concat([type_distribution, not_present_types_with_zeros]).sort_index()  # sort so order the same
            pit_v_bat_types_dist[i] = full_type_distribution   # store in array
    return pit_v_bat_types_dist

def add_batter_strike_tendencies(batter_specific_df, K_start, names):
    k_previous_types_dist = np.zeros((batter_specific_df.shape[0], len(names)))
    k_previous_types_dist[0:K_start, :] = np.nan
    
    for i in range(K_start, batter_specific_df.shape[0]):    # first pitch has no previous, store with nan
        slice = batter_specific_df.iloc[0:i,:] # get K previous pitches
        type_distribution = slice.type.value_counts(normalize=True)   # get rates each type thrown
        not_present_types = list(set(names) - set(type_distribution.index))  # get all the types not thrown by that pitcher within K pitches
        not_present_types_with_zeros = pd.Series(len(not_present_types)*[0.0], index = not_present_types)   # identify that these types were thrown 0% of the time
        full_type_distribution = pd.concat([type_distribution, not_present_types_with_zeros]).sort_index()  # sort so order the same
        k_previous_types_dist[i] = full_type_distribution   # store in array

    return pd.DataFrame(k_previous_types_dist, columns=names)

def main():
    SAVE = False
    pbp_filepath = "pitches_folder/pitch_by_pitch_metadata.csv"
    pitches_filepath = "pitches_folder/pitches"

    with open(pbp_filepath, 'rb') as rawdata:
        result = chardet.detect(rawdata.read(100000))

    pitch_by_pitch_metadata = pd.read_csv("pitches_folder/pitch_by_pitch_metadata.csv", encoding=result['encoding'])
    pitches = pd.read_csv("pitches_folder/pitches")

    pitch_types_to_keep = ['FF', 'FT', 'SI', 'FC', 'FS', 'SL', 'CU', 'CH', 'KC', 'KN', 'EP', 'FO', 'SC']    # 13 seems to be common
    pitches_modified = pitches.query("pitch_type in @pitch_types_to_keep").reset_index(drop=True)

    targets = ['pitch_type', 'type_confidence']
    games_features = ['inning', 'top', 'score_diff', 'at_bat_num', 'p_throws', 'pcount_pitcher', 'bases_state', 'pitcher_id']
    count_features = ['pcount_at_bat', 'balls', 'strikes', 'outs']
    batter_features = ['stand', 'height_inches']
    # new features that look at previous pitch of pitcher
    cols_for_previous = ['pitch_type', 'type', 'end_speed', 'break_length', 'break_angle', 'break_y', 'zone', 'spin_dir', 'spin_rate']
    cols_for_previous_names = ['prev_' + x for x in cols_for_previous]

    pitches_modified = convert_bases_features(pitches_modified)
    pitches_modified['bases_state'] = pitches_modified.on_1b + pitches_modified.on_2b + pitches_modified.on_3b 
    pitches_modified['bases_state'] = pitches_modified['bases_state'].astype('category') # Categories (8, object): ['000', '001', '010', '011', '100', '101', '110', '111']
    pitches_modified['is_home'] = pitches_modified['top']
    pitches_modified['score_diff'] = (pitches_modified.is_home)*(pitches_modified.home_team_runs - pitches_modified.away_team_runs) + (pitches_modified.is_home - 1)*(pitches_modified.home_team_runs - pitches_modified.away_team_runs)
    pitches_modified['height_inches'] = pitches_modified['b_height'].apply(lambda x : int(x.split("-")[0])*12 + int(x.split("-")[1]))
    pitches_modified[cols_for_previous_names] = pitches_modified.groupby(["pitcher_id", "game_pk", "at_bat_num"])[cols_for_previous].shift(1)

    dataset_cols_to_keep = ['uid'] + targets + games_features + count_features + batter_features + cols_for_previous_names
    dataset = convert_type(pitches_modified[dataset_cols_to_keep].copy(deep=True))
    if SAVE:
        dataset.to_pickle("data/dataset.pkl")

    ''' Advanced Dataset '''
    advanced_dataset = pitches_modified.sort_values(['pitcher_id', 'game_pk', 'pcount_pitcher']).reset_index(drop=True)
    all_type_names = pitches_modified.pitch_type.unique()
    strike_ball_hit_names = advanced_dataset.type.unique()
    # define new column names
    prev_5_col_names = [f'prev_5_' + f"{name}_%" for name in np.sort(all_type_names).tolist()]
    prev_10_col_names = [f'prev_10_' + f"{name}_%" for name in np.sort(all_type_names).tolist()]
    prev_20_col_names = [f'prev_20_' + f"{name}_%" for name in np.sort(all_type_names).tolist()]
    prev_5_col_names_strike = [f'prev_5_' + f"{name}_%" for name in np.sort(strike_ball_hit_names).tolist()]
    prev_10_col_names_strike = [f'prev_10_' + f"{name}_%" for name in np.sort(strike_ball_hit_names).tolist()]
    prev_20_col_names_strike = [f'prev_20_' + f"{name}_%" for name in np.sort(strike_ball_hit_names).tolist()]
    historical_col_names = [f'historical_' + f"{name}_%" for name in np.sort(all_type_names).tolist()]
    pit_v_bat_col_names = [f'pit_v_bat_' + f"{name}_%" for name in np.sort(all_type_names).tolist()]
    historical_batter_col_names = [f'historical_batter_' + f"{name}_%" for name in np.sort(strike_ball_hit_names).tolist()]

    # add previous pitch type tendencies
    for k in [5,10,20]:
        prev_k_col_names = [f'prev_{k}_' + f"{name}_%" for name in np.sort(all_type_names).tolist()]
        prev_k_new_cols = advanced_dataset.groupby("pitcher_id").apply(add_prev_pitch_type_tendencies, K=k, names=all_type_names)
        advanced_dataset.loc[:,prev_k_col_names] = prev_k_new_cols.values
    # add previous strike tendencies
    for k in [5,10,20]:
        prev_k_col_names = [f'prev_{k}_' + f"{name}_%" for name in np.sort(strike_ball_hit_names).tolist()]
        prev_k_new_cols = advanced_dataset.groupby("pitcher_id").apply(add_pitch_strike_tendencies, K=k, names=strike_ball_hit_names)
        advanced_dataset.loc[:,prev_k_col_names] = prev_k_new_cols.values
    # add historical tendencies
    historical_new_cols = advanced_dataset.groupby("pitcher_id").apply(add_pitcher_historical_tendencies, K_start=21, names=all_type_names)
    advanced_dataset.loc[:,historical_col_names] = historical_new_cols.values
    # add previous matchups
    advanced_dataset.loc[:,pit_v_bat_col_names] = add_pit_v_bat_matchups(advanced_dataset, names=all_type_names)
    # add batter tendencies. Unoptimized, do weird sorting before and after to make sure dataframe and outputted tendencies align
    advanced_dataset = advanced_dataset.sort_values(['batter_id', 'game_pk', 'pcount_at_bat']).reset_index(drop=True)
    historical_batter_new_cols = advanced_dataset.groupby("batter_id").apply(add_batter_strike_tendencies, K_start=21, names=strike_ball_hit_names)
    advanced_dataset.loc[:,historical_batter_col_names] = historical_batter_new_cols.values
    advanced_dataset = advanced_dataset.sort_values(['pitcher_id', 'game_pk', 'pcount_pitcher']).reset_index(drop=True)
    # define all columns to keep
    cols_to_keep = ['uid'] + targets + games_features + count_features + batter_features + cols_for_previous_names \
                    + prev_5_col_names + prev_10_col_names + prev_20_col_names + prev_5_col_names_strike + prev_10_col_names_strike \
                    + prev_20_col_names_strike + historical_col_names + pit_v_bat_col_names + historical_batter_col_names

    advanced_dataset_to_keep = convert_type(advanced_dataset[cols_to_keep].copy(deep=True))
    advanced_dataset_to_keep = advanced_dataset_to_keep.sort_values("uid") # so same order as dataset
    if SAVE:
        advanced_dataset_to_keep.to_pickle("data/advanced_dataset_final.pkl")

if __name__ == "__main__":
    main()