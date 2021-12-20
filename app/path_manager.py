import sys
import os

class Datapath:

    def __init__(self, data_dir):
        self.data_dir = data_dir


    def get_root_dir(self):
        '''
            returns the root directory of the data
        '''
        return self.data_dir


    def get_data_dir(self):
        '''
            returns the directory of the data
        '''
        return os.path.join(self.get_root_dir(), 'data')


    def get_data_raw_dir(self):
        '''
            returns the directory of the raw data
        '''
        return os.path.join(self.get_data_dir(), 'raw')


    def get_data_processed_dir(self):
        '''
            returns the directory of the processed data
        '''
        return os.path.join(self.get_data_dir(), 'processed')


    def get_resources_dir(self):
        '''
            returns the directory of the features
        '''
        return os.path.join(self.get_root_dir(), 'resources')


    def get_category_dir(self):
        '''
            returns the resources of category directory
        '''
        return os.path.join(self.get_resources_dir(), 'categories')


    def get_category_lexicon(self):
        '''
            returns the resources of category lexicon
        '''
        return os.path.join(self.get_category_dir(), 'categories.txt')


    def get_train_sentence_to_category_path(self, filename):
        '''
            returns the resources of sentence to category mapping of IBM watson api for training dataset
        '''
        return os.path.join(self.get_category_dir(), 'train/' + str(filename))


    def get_test_sentence_to_category_path(self, filename):
        '''
            returns the resources of sentence to category mapping of IBM watson api for testing dataset
        '''
        return os.path.join(self.get_category_dir(), 'test/' + str(filename))


    def get_eval_sentence_to_category_path(self, filename):
        '''
            returns the resources of sentence to category mapping of IBM watson api for evaluating dataset
        '''
        return os.path.join(self.get_category_dir(), 'eval/' + str(filename))


    def get_features_dir(self):
        '''
            returns the directory of the features
        '''
        return os.path.join(self.get_root_dir(), 'features')

    def get_features_quantiles_group_dir(self):
        '''
            returns the directory of the features' quantiles group
        '''
        return os.path.join(self.get_features_dir(), 'train/quantiles_group')


    def get_models_dir(self):
        '''
            returns the directory of the models
        '''
        return os.path.join(self.get_root_dir(), 'models')


    def get_results_dir(self):
        '''
            returns the directory of the results
        '''
        return os.path.join(self.get_root_dir(), 'results')


    def get_output_dir(self):
        '''
            returns the outputt directory
        '''
        return os.path.join(self.get_root_dir(), 'output')


    def get_visualization_dir(self):
        '''
            returns the directory of the visualization
        '''
        return os.path.join(self.get_root_dir(), 'visualization')

