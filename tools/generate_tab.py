import numpy as np
import pandas as pd
from tools.scores import generate_score
import pickle
import pandas as pd
from openpyxl import load_workbook

def generate_sampling_data(sampling_strategie, s_param, name_file):
    '''
    Take a sampling strategy as input, its parameters and a given data to generate an oversampled trainining part
    @params:
        - sampling_strategie: oversampling name
        - s_param: parameters of the oversampling strategy
        - name_file: name of the given data
    @return: oversampled train and its corresponding labels
    '''
    all_data = []
    for cv in range(5):
        sm = sampling_strategie(**s_param)
        a_file = open("./data/save_pickle/{0}_cv{1}.pkl".format(name_file, cv), "rb")
        output = pickle.load(a_file)
        train, test, train_labels, test_labels = output['train'], output['test'], output['train_labels'], output['test_labels']
        train, train_labels = sm.fit_sample(train, train_labels)
        all_data.append((train, train_labels, test, test_labels))
    return all_data

def generate_tab(sampling_strategies, score_strategies, name_file, list_classifiers):
    '''
    For a given sampling strategy, return the appropriate parameters that optimise the selected score.
    @params:
        - sampling_strategies: oversampling strategy
        - score_strategies: given score name
        - name_file: dataset name
        - list_classifiers: machine learning classifier to train the oversampled train part
    @return: dictionary containing the best parameters for the best couple (oversampling, classifier)
    '''
    dico_final = dict()
        
    for name_score in score_strategies:
        dico_final[name_score] = dict()
    
    for s_strategie in sampling_strategies:
  
        sampling_parameter = s_strategie().get_parameters_combination()
        
        couples = []
        nombre = 0
        dico_mean_sampling = dict()
        
        for name_score in score_strategies:
            dico_mean_sampling[name_score] = []
                
        for s_param in sampling_parameter:
          
            all_data = generate_sampling_data(s_strategie, s_param, name_file)
            
            for classifier in list_classifiers:
                couples.append((s_param, classifier))
                
                cv_ = dict()
                for train, train_labels, test, test_labels in all_data:
                    classifier.fit(train, train_labels)
                    predictions = classifier.predict(test)
                    predictions_proba = classifier.predict_proba(test)
                    for name_score in score_strategies:
                        if name_score not in cv_.keys():
                            cv_[name_score] = []
                        cv_[name_score].append(generate_score(name_score, test_labels, predictions, predictions_proba))
                
                for name_score in score_strategies:
                    cv_[name_score] = np.nan_to_num(cv_[name_score], copy=True, nan=-1, posinf=None, neginf=None)
                    dico_mean_sampling[name_score].append(np.mean(cv_[name_score]))
        
        for name_score in score_strategies:
            dico_final[name_score][s_strategie] = (couples[np.array(dico_mean_sampling[name_score]).argmax(axis=0)][0], couples[np.array(dico_mean_sampling[name_score]).argmax(axis=0)][1], couples[np.array(dico_mean_sampling[name_score]).argmax(axis=0)][1].get_params())
    
    return dico_final
                
def generate_tab_to_excel(sampling_strategies, score_strategies, name_file, list_classifiers):
    '''
    Write the final dictionnary containing parameters into an excel file.
    @params: 
        - sampling_strategies: oversampling strategies
        - score_strategies: given score
        - name_file: data file
        - list_classifiers: list of machine learning classifiers
    @return: None
    '''
    
    dico_params = generate_tab(sampling_strategies, score_strategies, name_file, list_classifiers)
    FilePath = 'C:/Users/oucht/Documents/These/Pipeline_Variants/parameters.xlsx'
    ExcelWorkbook = load_workbook(FilePath)
    writer = pd.ExcelWriter(FilePath, engine = 'openpyxl')
    writer.book = ExcelWorkbook

    for score_s in score_strategies:
        data = pd.DataFrame(dico_params[score_s])
        data.to_excel(writer, sheet_name = "{0}_{1}".format(score_s, name_file))

    writer.save()
    writer.close()

        