import pandas as pd
import numpy as np
import pickle
import logging
import json

import data_clean_for_model
import PipelineHelper

logger = logging.getLogger(__name__)

### 1. Load Data
df = pd.read_parquet("data/all_processed_df.parquet.gzip")

k = 5
rseed = 229
df["outcome"] = np.where( df["state"]=="successful", 1, 0, )
df["un_id"] = np.arange(0, df.shape[0], 1 )
df["name_len"] = df["name"].str.len()
df["cv_group"] = np.random.choice( np.arange(0, k), size=df.shape[0] )
df["binned_usd_goal"] = pd.qcut( np.log(df["usd_goal"]+1), 20 )

with open("model_config.json", 'r') as j:
     model_params = json.loads(j.read())
model_params['naive_bayes']['ngram_range'] = tuple(model_params['naive_bayes']['ngram_range'])


## load project metadata
logger.info("Loading features")
try:
    f = open("data/features.pkl", "rb")
    ft_dict = pickle.load(f)
    f.close()
    X_train, y_train, X_test, y_test = ft_dict.values()
except:
    X_train, X_test, y_train, y_test = data_clean_for_model.data_clean_for_model(df, "outcome", model_params, cv=model_params["cv"])

# load text
logger.info("Processing text data")
blurb_train, blurb_test, _, _    = data_clean_for_model.process_blurb(df, params)


## 2. Run text models

# get naive bayes predictions
logger.info("Loading Naive Bayes predictions")
try:
    nb_proba_train = np.load("data/res/multi_nb_preds_train.npy")
    nb_proba_test = np.load("data/res/multi_nb_preds_test.npy")
except:
    logger.info("Running Naive Bayes model")
    nb_params = model_params['naive_bayes']
    nb_train_pred, nb_proba_train, nb_test_pred, nb_proba_test = PipelineHelper.naive_bayes_predictions(
        blurb_train, y_train, blurb_test,
        tfidf=nb_params['tf-idf'], ngram_range=nb_params['ngram_range']
    )
    np.save("data/res/multi_nb_preds_train.npy", nb_proba_train)
    np.save("data/res/multi_nb_preds_test.npy", nb_proba_test)

# get LDA topic model
logger.info("Loading LDA topic predictions")
try:
    lda_train = pd.read_csv("data/res/lda_train.csv").drop(columns=['Unnamed: 0'])
    lda_test = pd.read_csv("data/res/lda_test.csv").drop(columns=['Unnamed: 0'])
except:
    logger.info("Running LDA topic model")
    lda_params = model_params['lda_params']
    tokenized_train = blurb_train.apply(data_clean_for_model.tokenize_text)
    tokenized_test = blurb_test.apply(data_clean_for_model.tokenize_text)
    lda_train, lda_test = PipelineHelper.train_lda_model(tokenized_train, tokenized_test, params['lda'])
    lda_train.to_csv("data/res/lda_train.csv")
    lda_test.to_csv("data/res/lda_test.csv")

# get Word2Vec model predictions
logger.info("Loading Word2Vec dimension predictions")
try:
    w2v_train = np.load("data/res/w2v_Xtrain_avg_big.npy")
    w2v_test = np.load("data/res/w2v_Xtest_avg_big.npy")
except:
    raise Warning("Word2Vec function not implemented. Running without it -- likely will crash.")


## 3. Analyses

### a. Just on metadata
logger.info("Getting metadata results")
stat_df, pred_df, models = PipelineHelper.run_analyses(X_train, y_train, X_test, y_test, model_params)

### b. Just on metadata, - binned_usd_goal_outcome_mean
logger.info("Getting metadata - binned_usd_goal_outcome_mean results")
stat_df_nobinusd, pred_df_nobinusd, models_nobinusd = PipelineHelper.run_analyses(
    X_train.drop(columns=['binned_usd_goal_outcome_mean']), y_train,
    X_test.drop(columns=['binned_usd_goal_outcome_mean']), y_test, model_params
)

### c. Just on metadata + nb
logger.info("Getting metadata + naive bayes results")
X_train_nb = X_train.copy()
X_test_nb = X_test.copy()
X_train_nb['nb_proba'] = nb_proba_train[:, 1]
X_test_nb['nb_proba'] = nb_proba_test[:, 1]
stat_df_nb, pred_df_nb, models_nb = PipelineHelper.run_analyses(X_train_nb, y_train, X_test_nb, y_test, model_params)

### d. Just on metadata + nb + lda
logger.info("Getting metadata + naive bayes + LDA results")
X_train_nb_lda = pd.concat((X_train_nb, lda_train), axis=1)
X_test_nb_lda = pd.concat((X_test_nb, lda_test), axis=1)
stat_df_nb_lda, pred_df_nb_lda, models_nb_lda = PipelineHelper.run_analyses(X_train_nb_lda, y_train, X_test_nb_lda, y_test, model_params)

### e. Just on metadata + nb + w2v
logger.info("Getting metadata + naive bayes + w2v results")
X_train_nb_w2v = pd.concat((X_train_nb, pd.DataFrame(w2v_train)), axis=1)
X_test_nb_w2v = pd.concat((X_test_nb, pd.DataFrame(w2v_test)), axis=1)
stat_df_nb_w2v, pred_df_nb_w2v, models_nb_w2v = PipelineHelper.run_analyses(X_train_nb_w2v, y_train, X_test_nb_w2v, y_test, model_params)

### f. Just on metadata + nb + lda - cols to drop
logger.info("Getting metadata + naive bayes + LDA results")
cols_to_drop = [
    'dummy_cat_id_290', 'dummy_cat_id_300', 'dummy_cat_id_317','dummy_cat_id_386', 'dummy_cat_id_1', 'dummy_cat_id_352',
    'dummy_cat_id_355', 'dummy_cat_id_354', 'dummy_cat_id_321', 'dummy_cat_id_12', 'dummy_cat_id_340', 'dummy_cat_id_268', 'binned_usd_goal_outcome_mean'
]
stat_df_nb_lda_drop, pred_df_nb_lda_drop, models_nb_lda_drop = PipelineHelper.run_analyses(
    X_train_nb_lda.drop(columns=cols_to_drop), y_train,
    X_test_nb_lda.drop(columns=cols_to_drop), y_test, model_params
)

## 4. Results
# specify data
logger.info("Creating model statistics df")
stat_df.insert(0, "data", "metadata"),
stat_df_nobinusd.insert(0, "data", "metadata_nobin"),
stat_df_nb.insert(0, "data", "metadata_nb"),
stat_df_nb_lda.insert(0, "data", "metadata_nb_lda"),
stat_df_nb_w2v.insert(0, "data", "metadata_nb_w2v"),
stat_df_nb_lda_drop.insert(0, "data", "metadata_nb_lda_drop")
fin = (pd.concat((stat_df, stat_df_nobinusd, stat_df_nb, stat_df_nb_lda, stat_df_nb_w2v, stat_df_nb_lda_drop))
       .sort_values('accuracy', ascending=False)
       .assign(
           accuracy_rank = lambda x:np.arange(1, x.shape[0]+1, 1),
           random_state = model_params['rseed']
       )
      )
fin.to_csv("model_exports/model_statistics.csv")
logger.info("Creating model predictions df")
pred_df.columns = "metadata_" + pred_df.columns
pred_df_nobinusd.columns = "metadata_nobin_" + pred_df_nobinusd.columns
pred_df_nb.columns = "metadata_nb_" + pred_df_nb.columns
pred_df_nb_lda.columns = "metadata_nb_lda_" + pred_df_nb_lda.columns
pred_df_nb_w2v.columns = "metadata_nb_w2v_" + pred_df_nb_w2v.columns
pred_df_nb_lda_drop.columns = "metadata_nb_lda_drop_" + pred_df_nb_lda_drop.columns
pred_fin = pd.concat((pred_df, pred_df_nobinusd, pred_df_nb, pred_df_nb_lda, pred_df_nb_w2v, pred_df_nb_lda_drop), axis=1)
pred_fin.to_csv("model_exports/model_predictions.csv")
