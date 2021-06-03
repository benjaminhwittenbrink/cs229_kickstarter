import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def data_clean_for_model(df, outcome, params, cv=True):

    df = add_features(df)
    if cv:
        X_train, X_test, y_train, y_test = train_test_split(df, df[outcome], test_size=params['test_frac'], random_state=params['rseed'])
        X_train_lnom, X_test_lnom = encode_categoricals(X_train, X_train, X_test, add_cv_lnoms, params)
    else:
        X_train, X_lnom, X_test, y_train, y_lnom, y_test = split_data(df, outcome, params['lnom_frac'], params['test_frac'], params['rseed'])
        X_train_lnom, X_test_lnom = encode_categoricals(X_train, X_lnom, X_test, add_lnoms, params)

    # specify feature set here
    X_cols_all = specify_features(X_train_lnom, lnom_usdgoal=params['lnom_usdgoal'], dummies=params['dummies'])
    # subset features on our train and test data
    X_train_lnom = X_train_lnom[X_cols_all]
    X_test_lnom = X_test_lnom[X_cols_all]

    # make sure all values are floats, if na --> fill w/ 0.5 (i.e. we have no better guess)
    # there aren't many nas though, only really in cat_parent_id and loc_type
    if cv:
        X_test_lnom = replace_nas(X_test_lnom, X_train, X_cols_all)
        X_train_lnom = replace_nas(X_train_lnom, X_train, X_cols_all)
    else:
        X_test_lnom = replace_nas(X_test_lnom, X_lnom, X_cols_all)
        X_train_lnom = replace_nas(X_train_lnom, X_lnom, X_cols_all)
    ## subtract actual usd_goal from category goal
    if params['lnom_usdgoal']:
        X_lnom_cols_usdgoal = X_train_lnom.columns[ np.logical_and(X_train_lnom.columns.str.contains("usd_goal_mean"), np.logical_not(X_train_lnom.columns.str.contains("binned"))) ]
        X_lnom_cols_usdgoal_sub = [i + "_sub" for i in X_lnom_cols_usdgoal]
        X_train_lnom[X_lnom_cols_usdgoal_sub] = np.log(X_train_lnom[X_lnom_cols_usdgoal]+1).sub(np.log(X_train_lnom['usd_goal']+1), axis= 0 )
        X_test_lnom[X_lnom_cols_usdgoal_sub] =  np.log(X_test_lnom[X_lnom_cols_usdgoal]+1).sub(np.log(X_test_lnom['usd_goal']+1), axis= 0 )

    return X_train_lnom, X_test_lnom, y_train, y_test

def add_features(df):
    df = df.assign(
        time_diff = lambda x:x['deadline']-x['launched_at']
    )
    return df

def split_data(df, outcome, lnom_frac, test_frac, rseed):

    X_train, X_lnom, y_train, y_lnom = train_test_split(df, df[outcome], test_size=lnom_frac, random_state=rseed)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=test_frac, random_state=rseed)

    return X_train, X_lnom, X_test, y_train, y_lnom, y_test

def specify_features(X_train, lnom_usdgoal=True, dummies=False):
    cv_col = ["cv_group", "un_id"]
    # categorical variables (we won't use these raw, but will use lnom results)
    categorical_cols = ["currency", "country", "cat_id", "cat_parent_id", "loc_id", "loc_type"]
    categorical_cols_stds = ["cat_id", "cat_parent_id"]
    # for results for lnom
    X_lnom_cols_outcome = pd.Series(categorical_cols + ["binned_usd_goal"]) + "_outcome_mean"
    X_lnom_cols_usdgoal = pd.Series(categorical_cols) + "_usd_goal_mean"
    X_std_cols = np.concatenate((pd.Series(categorical_cols_stds) + "_outcome_std", pd.Series(categorical_cols_stds) + "_usd_goal_std"))
    # specify other features here
    X_cols = [ "blurb_len", "name_len", "usd_goal", "deadline", "launched_at", "time_diff"]

    if lnom_usdgoal: X_cols_all = np.concatenate((cv_col, X_cols, X_lnom_cols_outcome, X_lnom_cols_usdgoal, X_std_cols))
    else: X_cols_all = np.concatenate((cv_col, X_cols, X_lnom_cols_outcome, X_std_cols))
    if dummies:
        dummy_cols = [col for col in X_train.columns if "dummy" in col]
        X_cols_all = np.concatenate((X_cols_all, dummy_cols))
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

def encode_categoricals(X_train, X_lnom, X_test, add_lnoms_func, params):
    # add outcome lnoms
    X_train_lnom = add_lnoms_func(X_train, X_lnom, "outcome")
    X_test_lnom = add_lnoms_func(X_test, X_lnom, "outcome")
    if params['cv']:
        X_train_lnom = add_cv_lno_stds(X_train_lnom, X_lnom, "outcome")
        X_test_lnom = add_cv_lno_stds(X_test_lnom, X_lnom, "outcome")
    # add usd_goal lnoms (let's ignore these for now --> don't help performance at all, if anything hurt it )
    if params['lnom_usdgoal']:
        X_train_lnom = add_lnoms_func(X_train_lnom, X_lnom, "usd_goal")
        X_test_lnom = add_lnoms_func(X_test_lnom, X_lnom, "usd_goal")
        if params['cv']:
            X_train_lnom = add_cv_lno_stds(X_train_lnom, X_lnom, "usd_goal")
            X_test_lnom = add_cv_lno_stds(X_test_lnom, X_lnom, "usd_goal")
    if params['dummies']:
        dummy_cols = ["cat_id", "cat_parent_id"]
        X_train_lnom = pd.get_dummies(X_train_lnom, columns=dummy_cols, prefix=["dummy_"+i for i in dummy_cols])
        X_test_lnom = pd.get_dummies(X_test_lnom, columns=dummy_cols, prefix=["dummy_"+i for i in dummy_cols])
    return X_train_lnom, X_test_lnom

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
    ).merge(
        lnom(df2, "binned_usd_goal", target_col), on="binned_usd_goal", how="left"
    )
    return dfs_lnom

def cv_lnom(df, grp_col, target_col="outcome", min_n=1000):
    k = df['cv_group'].max()+1
    out_col = grp_col + "_" + target_col + "_mean"
    tmp_df = df.groupby([grp_col, "cv_group"])[target_col].agg(
        [(out_col, "sum"), ("c", "count")]
    ).reset_index()
    tmp_df = pd.concat( (tmp_df.assign(cv_main = i) for i in range(k) ))
    tmp_df = tmp_df.loc[ tmp_df['cv_main'] != tmp_df['cv_group'] ]
    res = tmp_df.groupby([grp_col, "cv_main"]).agg('sum')
    res[out_col] = res[out_col]/res['c']
    res = res.loc[res['c']>=min_n].drop(columns=['cv_group']).reset_index().rename(columns={'cv_main':'cv_group'})
    return res[[grp_col, "cv_group", out_col]]

def add_cv_lnoms(df1, df2, target_col="outcome"):
    """
    lnom = "Leave none out mean", i.e. calculate the average within a group. (as opposed to a loom -- "leave one out of mean")
    Add lnoms for a bunch of categorical variables (currently hard-coded).
    Lnoms are calculated using df2 and then merged onto df1 (i.e. insuring indepdence, though don't think this all that important).
    Target column is specified with target_col.
    """
    dfs_lnom = df1.merge(
        cv_lnom(df2, "currency", target_col), on=["currency", "cv_group"], how="left"
    ).merge(
        cv_lnom(df2, "country", target_col), on=["country", "cv_group"], how="left"
    ).merge(
        cv_lnom(df2, "cat_id", target_col), on=["cat_id", "cv_group"], how="left"
    ).merge(
        cv_lnom(df2, "cat_parent_id", target_col), on=["cat_parent_id", "cv_group"], how="left"
    ).merge(
        cv_lnom(df2, "loc_type", target_col), on=["loc_type", "cv_group"], how="left"
    ).merge(
        cv_lnom(df2, "loc_id", target_col), on=["loc_id", "cv_group"], how="left"
    ).merge(
        cv_lnom(df2, "binned_usd_goal", target_col), on=["binned_usd_goal", "cv_group"], how="left"
    )
    return dfs_lnom

def cv_lno_std(df, grp_col, target_col="outcome", min_n=1000):
    k = df['cv_group'].max()+1
    out_col = grp_col + "_" + target_col + "_std"
    tmp_df = pd.concat( (df.assign(cv_main = i) for i in range(k) ))
    tmp_df = tmp_df.loc[ tmp_df['cv_main'] != tmp_df['cv_group'] ]
    res = tmp_df.groupby([grp_col, "cv_main"])[target_col].agg(
        [(out_col,"std"), ("c", "count")]
    )
    res = res.loc[res['c']>=min_n].reset_index().rename(columns={'cv_main':'cv_group'})
    return res[[grp_col, "cv_group", out_col]]

def add_cv_lno_stds(df1, df2, target_col="outcome"):
    dfs_lno_std = df1.merge(
        cv_lno_std(df2, "cat_id", target_col), on=["cat_id", "cv_group"], how="left"
    ).merge(
        cv_lno_std(df2, "cat_parent_id", target_col), on=["cat_parent_id", "cv_group"], how="left"
    ).merge(
        cv_lno_std(df2, "binned_usd_goal", target_col), on=["binned_usd_goal", "cv_group"], how="left"
    )
    return dfs_lno_std



########
# Text Processing Methods
import re
import nltk
from nltk.corpus import stopwords
# consts
RE_replace_space = re.compile('[/(){}\[\]\|@,;]')
RE_symbols_to_drop = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))

def process_blurb(df, params):
    df_blurb_cln = df['blurb'].apply(clean_text)
    y = np.where(df['state'] =='successful', 1, 0)
    X_train, X_test, y_train, y_test = train_test_split(df_blurb_cln, y, test_size=params['test_frac'], random_state=params['rseed'])
    return X_train, X_test, y_train, y_test

def clean_text(txt):
    if txt is None: return ''
    txt = txt.lower()
    txt = RE_replace_space.sub(' ', txt)
    txt = RE_symbols_to_drop.sub('', txt)
    txt = ' '.join(word for word in txt.split() if word not in STOPWORDS)
    return txt

def tokenize_text(blurb_col):
    tokens = []
    for sent in nltk.sent_tokenize(blurb_col, language='english'):
        for word in nltk.word_tokenize(sent, language='english'):
            if len(word) >= 2: tokens.append(word)
    return tokens
