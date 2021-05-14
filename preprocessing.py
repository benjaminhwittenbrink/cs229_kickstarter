import pandas as pd
import numpy as np
import json
import os

from tqdm import tqdm

def process_data_all(folder_loc, folders, main_file):
    """
    Iteretes over all specified folders.
    """
    data = []
    for fold in folders:
        folder_path = folder_loc + fold
        files = sorted(os.listdir(folder_path))
        df = process_data_folder(fold, folder_path, files, save=True)
        print("Finished proccessing file {}".format(fold))
    # remove duplicate rows
    folders.remove(main_file)
    final_df = unduplicate_data(folder_path, main_file, folders)
    return final_df

def unduplicate_data(folder_path, main_file, addtl_files, features_dup=None, extension=".parquet.gzip"):
    """
    Removes duplicate projects from the list of dataframes.
    Each dataframe does not include just the unique projects of that month-year combination, but rather all
    projects the web robot finds when scraping Kickstarter. As a result, each dataframe consists of heavy
    overlap. Ideally, we would remove these rows earlier in the pipeline, but for now this seems fine.
    Currently, to drop duplicates we are selecting the following subset of features to identify :

                    ["blurb", "name", "usd_goal", "launched_at", "deadline"]

    types:
        main_file --> string to path of main file (name, no extension necessary)
        addtl_files --> list of string to path of addtl files to add (name, no extension necessary)
        features_dup --> list of features to use to identify duplicate rows
    """
    # in order to remove duplicate rows, we need to start with a main file
    main_data = pd.read_parquet(main_file + extension)
    # initialize duplicate features
    if features_dup is None: features_dup = ["blurb", "name", "usd_goal", "launched_at", "deadline"]
    for new_file in addtl_files:
        tmp = pd.read_parquet(new_file + extension)
        # combine and drop duplicates
        main_data = pd.concat((main_data, tmp)).drop_duplicates(subset=features_dup)
        del tmp # not sure if this is fully necessary, but ease load
        print("Adding file", new_file, "to yield new shape {}".format(main_data.shape))
    main_data.to_parquet(folder_path + "/" +  "all_processed_df" + extension)
    return main_data

def process_data_folder(folder_name, file_loc, file_paths, save=False):
    """
    Iterates over all csvs in given folder.
    """
    dfs = []
    for f in tqdm(file_paths):
        if ".csv" in f:
            dfs.append( process_data_csv( file_loc + "/" +  f ) )
        else:
            print("Error invalid file {}".format(f))
            print("Skipping to next.")
    fulldf = pd.concat(dfs)
    if save: fulldf.to_parquet(file_loc + "/" + folder_name + ".parquet.gzip", compression="gzip")
    return fulldf


def process_data_csv(file_path):
    """
    Processes data given a CSV file path.
    Adds some features, unpacks category and location information that is stored as json.
    """
    df = pd.read_csv(file_path)
    # filter rows to finished projects, i.e. success or failure
    df = df.loc[df['state'].isin(['successful', 'failed'])]
    ## now load some categories that are stored as jsons
    # add some features
    df = df.assign(
        usd_goal = lambda x: x["goal"] * x["fx_rate"],
        available_time = lambda x: x["deadline"] - x["launched_at"], # figure out how time is encoded?
        blurb_len = lambda x: x["blurb"].str.len()
    )
    # 1. project category
    cat_cols_to_keep = ["id", "position", "parent_id", "color"]
    tmp = df["category"].apply(json.loads).apply(pd.Series)[cat_cols_to_keep]
    tmp.columns = "cat_" + tmp.columns
    cat_cols_to_keep = ["cat_" + w for w in cat_cols_to_keep]
    df = pd.concat((df, tmp), axis = 1)

    # 2. project location
    loc_cols_to_keep = ["id", "type", "state"]
    tmp = df['location'].fillna('{}').apply(json.loads).apply(pd.Series)[loc_cols_to_keep]
    tmp.columns = "loc_" + tmp.columns
    loc_cols_to_keep = ["loc_" + w for w in loc_cols_to_keep]
    df = pd.concat((df, tmp), axis=1)
    # 3. ??

    # specify cols to keep
    df_cols_to_keep = [
         "state", "usd_goal", "available_time", "blurb_len", "launched_at", "deadline", "blurb",
        "name", "currency", "country", "is_starred", "is_starrable", "spotlight", "staff_pick", "photo"
    ]
    return df[ df_cols_to_keep + cat_cols_to_keep + loc_cols_to_keep ]




if __name__ == '__main__':
    folders = ["2021_04", "2021_03", "2021_02", "2020_12", "2020_04"]
    process_data_all("data/", folders, "2021_04")
