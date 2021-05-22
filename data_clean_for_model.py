
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def data_clean_for_model(df, outcome, params):

    #X_train, X_test, y_train, y_test = train_test_split(dfs, dfs["outcome"], test_size=0.3, random_state=rseed)
    # to avoid info leakage, we'll calculate the averages on a different df
    X_train, X_lnom, X_test, y_train, y_lnom, y_test = split_data(df, outcome, params['lnom_frac'], params['test_frac'], params['rseed'])
    # add outcome lnoms
    X_train_lnom = add_lnoms(X_train, X_lnom, "outcome")
    X_test_lnom = add_lnoms(X_test, X_lnom, "outcome")
    # add usd_goal lnoms (let's ignore these for now --> don't help performance at all, if anything hurt it )
    if params['lnom_usdgoal']:
        X_train_lnom = add_lnoms(X_train_lnom, X_lnom, "usd_goal")
        X_test_lnom = add_lnoms(X_test_lnom, X_lnom, "usd_goal")
    # specify feature set here
    X_cols_all = specify_features(lnom_usdgoal=params['lnom_usdgoal'])
    # subset features on our train and test data
    X_train_lnom = X_train_lnom[X_cols_all]
    X_test_lnom = X_test_lnom[X_cols_all]
    # make sure all values are floats, if na --> fill w/ 0.5 (i.e. we have no better guess)
    # there aren't many nas though, only really in cat_parent_id and loc_type
    X_test_lnom = replace_nas(X_test_lnom, X_lnom, X_cols_all)
    X_train_lnom = replace_nas(X_train_lnom, X_lnom, X_cols_all)
    ## subtract actual usd_goal from category goal
    if params['lnom_usdgoal']:
        X_lnom_cols_usdgoal = X_train_lnom.columns[ X_train_lnom.columns.str.contains("usd_goal_mean") ]
        X_train_lnom[X_lnom_cols_usdgoal] = np.log(X_train_lnom[X_lnom_cols_usdgoal]+1).sub(np.log(X_train_lnom['usd_goal']+1), axis= 0 )
        X_test_lnom[X_lnom_cols_usdgoal] =  np.log(X_test_lnom[X_lnom_cols_usdgoal]+1).sub(np.log(X_test_lnom['usd_goal']+1), axis= 0 )

    return X_train_lnom, X_test_lnom, y_train, y_test

def split_data(df, outcome, lnom_frac, test_frac, rseed):

    X_train, X_lnom, y_train, y_lnom = train_test_split(df, df["outcome"], test_size=lnom_frac, random_state=rseed)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=test_frac, random_state=rseed)

    return X_train, X_lnom, X_test, y_train, y_lnom, y_test

def specify_features(lnom_usdgoal=True):

    # categorical variables (we won't use these raw, but will use lnom results)
    categorical_cols = ["currency", "country", "cat_id", "cat_parent_id", "loc_id", "loc_type"]
    # for results for lnom
    X_lnom_cols_outcome = pd.Series(categorical_cols) + "_outcome_mean"
    X_lnom_cols_usdgoal = pd.Series(categorical_cols) + "_usd_goal_mean"
    # specify other features here
    X_cols = ["blurb_len", "name_len", "usd_goal", "deadline", "launched_at"]

    if lnom_usdgoal: X_cols_all = np.concatenate((X_cols, X_lnom_cols_outcome, X_lnom_cols_usdgoal))
    else: X_cols_all = np.concatenate((X_cols, X_lnom_cols_outcome))

    return X_cols_all

def replace_nas(df1, df2, df1_varlist, lnom_usdgoal=True):
    """
    Impute NAs in our data (df1) with the overall means (best-guess) from a independent dataframe (df2).
    """
    # if grouped outcome means are missing, replace w/ over all outcome mean
    df1.loc[:, df1.columns.str.contains('outcome') ]  = df1.loc[:, df1.columns.str.contains('outcome')].fillna(  df2['outcome'].mean()  )

    # if using usd_goal variables
    if lnom_usdgoal:
        df1.loc[:, df1.columns.str.contains('usd_goal') ] = df1.loc[:, df1.columns.str.contains('usd_goal')].fillna( df2['usd_goal'].mean() )

    # if blurb len is missing, either: blurb is missing because it doesn't exist or it couldn't be scraped
    # to be more conservative Im just assuming the later and setting equal to blurb mean
    df1['blurb_len'] = df1['blurb_len'].fillna( df2['blurb_len'].mean() )
    return df1

def lnom(df, grp_col, target_col="outcome", min_n=1000):
    """
    Encode our categorical variables, by calculating the leave-nothing-out mean,
    i.e. for each category, we calculate the mean of the target and use it as our feature
    """
    out_col = grp_col + "_" + target_col + "_mean"
    tmp_df = df.groupby([grp_col])[target_col].agg(
        [(out_col, "mean"), ("c", "count")]
    )
    tmp_df = tmp_df.loc[tmp_df["c"] >= min_n].reset_index()
    return tmp_df[[grp_col, out_col]]

def add_lnoms(df1, df2, target_col="outcome"):
    """
    lnom = "Leave none out mean", i.e. calculate the average within a group. (as opposed to a loom -- "leave one out of mean")
    Add lnoms for a bunch of categorical variables (currently hard-coded).
    Lnoms are calculated using df2 and then merged onto df1 (i.e. insuring indepdence, though don't think this all that important).
    Target column is specified with target_col.
    """
    dfs_lnom = df1.merge(
        lnom(df2, "currency", target_col), on="currency", how="left"
    ).merge(
        lnom(df2, "country", target_col), on="country", how="left"
    ).merge(
        lnom(df2, "cat_id", target_col), on="cat_id", how="left"
    ).merge(
        lnom(df2, "cat_parent_id", target_col), on="cat_parent_id", how="left"
    ).merge(
        lnom(df2, "loc_type", target_col), on="loc_type", how="left"
    ).merge(
        lnom(df2, "loc_id", target_col), on="loc_id", how="left"
    )
    return dfs_lnom