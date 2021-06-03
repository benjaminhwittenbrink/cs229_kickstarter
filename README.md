# cs229_kickstarter

Sorry the code is very messy. 

The main pipeline is as follows: 
    1. preprocessing.py cleans the data as it is downloaded from Kickstarter Web Robots website. 
    2. pipeline.py and PipelineHelper.py structure the rest of the pipeline: 
        i). running data_clean_for_model.py which creates the categorical encodings and does most of the non text-based feature engineering 
        ii). runs naive bayes classifier and LDA  + W2V model to create text encodings
        iii). runs all our models on various sample definitions, ASIDE FROM
        iv). to obtain neural net predictions you need to have already run neural_net.ipynb and saved the model as a tf.keras.model object because it is very slow training 
        v). computes performance of all of these methods using model_metrics.py
        
NOTE: pipeline.py is currently a bit outdated relative to pipeline_notebook.ipynb. When there exist discrepancies, defer to the notebook. 
NOTE: A lot of the model training, cross-validation, etc. was done in various notebooks in dev and/or main analysis        
NOTE: Resolving conflict between Sauren's CoreNLP methods and my pipeline proved very cumbersome, so we just created two versions: 
saruen_pipeline_notebook and pipeline_notebook