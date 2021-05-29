import numpy as np
import pandas as pd
import nltk
import gensim
import logging

from model_metrics import format_results

from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer

import sklearn.linear_model as lm
import sklearn.ensemble as em
import lightgbm as lgb

logger = logging.getLogger(__name__)

def naive_bayes_predictions(X_train, y_train, X_test, tfidf=True, ngram_range=(1, 1)):
    """
    Fit a Multinomial Naive Bayes model and return the predictions on the train and test set.
    X_train & X_test --> vectors of string. Note: this function handles the tokenization and counting of word occurrences.
    tfidf: boolean vector whether to use a tf-idf transformer on word counts
    ngram_range: range of grams to use in the count vectorizer, e.g. (1,3) implies unigrams, bigrams, and trigrams,
    """
    if tfidf:
        nb = Pipeline([
            ('vect', CountVectorizer( ngram_range=ngram_range )),
            ('tfidf', TfidfTransformer()),
            ('clf', MultinomialNB())
        ])
    else:
        nb = Pipeline([
            ('vect', CountVectorizer( ngram_range=ngram_range )),
            ('clf', MultinomialNB())
        ])
    # fit naive bayes
    nb.fit(X_train, y_train)
    # get predictions
    y_pred_train = np.predict(X_train)
    y_pred_test = nb.predict(X_test)
    y_pred_train_proba = nb.predict_proba(X_train)
    y_pred_test_proba = nb.predict_proba(X_test)
    return y_pred_train, y_pred_train_proba, y_pred_test, y_pred_test_proba



def train_lda_model(train_token, test_token, params):

    def convert_to_bigram(words, bi_min=10):
        gram = gensim.models.Phrases(words, min_count=bi_min)
        return gensim.models.phrases.Phraser(gram)

    def get_corpus(words):
        # construct bigrams
        bigram = convert_to_bigram(words)
        bigram = [bigram[r] for r in words]
        # construct mapping from id to bigram
        id2word = gensim.corpora.Dictionary(bigram)
        id2word.filter_extremes(no_below=params['corpus']['no_below'], no_above=params['corpus']['no_above'])
        id2word.compactify()
        # create corpus
        corpus = [id2word.doc2bow(text) for text in bigram]
        return corpus, id2word, bigram

    def get_lda_predictions(tokens, corpus):
        vecs = []
        for i in range(len(tokens)):
            # get topic vector probabilities for every example
            top_topics = lda_model.get_document_topics(corpus[i], minimum_probability=0)
            topic_vec = [top_topics[i][1] for i in range(params['n_topics'])]
            vecs.append(topic_vec)
        df = pd.DataFrame(vecs)
        df.columns = ["lda_topic" + str(i) for i in df.columns]
        return df

    # get train corpus + bigrams
    corpus_train, id2word_train, bigram_train = get_corpus(train_token)
    # get test corpus + bigrams
    bigram_test = convert_to_bigram(test_token)
    bigram_test = [bigram_test[r] for r in test_token]
    corpus_test = [id2word_train.doc2bow(text) for text in bigram_test]
    # specify LDA model
    lda_model = gensim.models.ldamulticore.LdaMulticore(
        corpus=corpus_train, num_topics=params['n_topics'],
        id2word=id2word_train, chunksize=params['chunksize'],
        passes=params['passes'], eval_every=1, per_word_topics=True,
        random_state=params['rseed']
    )
    # get predictions
    lda_train = get_lda_predictions(train_token, corpus_train)
    lda_test = get_lda_predictions(test_token, corpus_test)

    return lda_train, lda_test











def run_analyses(X_train, y_train, X_test, y_test, params):
    """
    Run a set of classification models according to passed in parameters.
    Fit on train data, predict on test data. Returns a dataframe of performance
    statistics (accuracy, f1, etc.) as well as a dataframe of classifications, and the model objects.
    """
    # get training weights (if function doesn't have a balanced parameter)
    train_weights, test_weights = get_weights(y_train, y_test)
    # fit linear models
    logger.info("Fitting linear models")
    linear_models = run_linear_models(X_train, y_train, train_weights, params['linear_models'], rseed=params['rseed'])
    stat_df_lm, pred_df_lm = format_results(linear_models, X_test, y_test)
    # fit ensemble methods -- lgbm + random forest right now
    # TO DO: add neural net??
    logger.info("Fitting lightgbm")
    lgbm = run_lgb(X_train, y_train, params['lightgbm'], rseed=params['rseed'])
    logger.info("Fitting random forest")
    rf = run_rf(X_train, y_train, params['random_forest'], rseed=params['rseed'])
    stat_df_em, pred_df_em = format_results([lgbm, rf], X_test, y_test, ["LGBMClassifier", "RandomForestClassifier"])
    # concatenate results
    stat_df = pd.concat((stat_df_lm, stat_df_em))
    pred_df = pd.concat((pred_df_lm, pred_df_em), axis=1)
    models = linear_models + [lgbm, rf]
    return stat_df, pred_df, models

def get_weights(y_train, y_test):
    """
    Calculate weights to make the train and test sample balanced by class.
    """
    N0_train = (y_train==0).sum()
    N1_train = (y_train==1).sum()
    N0_test = (y_test==0).sum()
    N1_test = (y_test==1).sum()
    train_weights = np.where(y_train==1, 0.5/N1_train, 0.5/N0_train)
    test_weights = np.where(y_test==1, 0.5/N1_test, 0.5/N0_test)
    return train_weights, test_weights


def run_linear_models(X_train, y_train, train_weights, params, rseed=229):
    """
    Run linear models: OLS, Lasso, Ridge, and Logistic Regression.
    """
    ### a. OLS
    ols = lm.LinearRegression(normalize=False) # drop cols below to avoid perfect colinearity
    perf_col_dummies = ['dummy_cat_id_1','dummy_cat_parent_id_1.0']
    ols.fit(X_train.drop(columns=perf_col_dummies, errors='ignore'), y_train, sample_weight=train_weights)
    else: ols.fit(X_train.drop(columns=perf_col_dummies), y_train, sample_weight=train_weights)
    ### b. Lasso
    clf_lasso = lm.Lasso(alpha=params['lasso_alpha'], normalize=False, random_state=rseed)
    clf_lasso.fit(X_train, y_train, sample_weight=train_weights)
    ### c. Ridge
    clf_ridge = lm.Ridge(alpha=params['ridge_alpha'], normalize=False, random_state=rseed)
    clf_ridge.fit(X_train, y_train, sample_weight=train_weights)
    ### d. Logistic
    logreg = lm.LogisticRegression(C=params['logreg_C'], penalty=params['logreg_penalty'], random_state=rseed, class_weight='balanced')
    logreg.fit(X_train, y_train)

    return [ols, clf_lasso, clf_ridge, logreg]


def run_lgb(X_train, y_train, params, rseed=229):
    """
    Wrapper for lightgbm classification sklearn API.
    """
    lgbm = lgb.LGBMClassifier(random_state=rseed, is_unbalance=True, verbose=-1, **params)
    lgbm.fit(X_train, y_train)
    return lgbm

def run_rf(X_train, y_train, params, rseed=229):
    """
    Wrapper for random forest sklearn API.
    """
    rf = em.RandomForestClassifier(random_state=rseed, class_weight='balanced', **params)
    rf.fit(X_train, y_train)
    return rf
