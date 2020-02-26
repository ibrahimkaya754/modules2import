# Helper Functions

from import_modules import *

class split_data:
    def __init__(self,features_input,target_output,scaler,shuffle=True):

        self.features_input    = features_input
        self.target_output     = target_output
        self.scaler            = scaler
        self.shuffle           = shuffle
        self.counter           = 0
        self.namekey           = []

    def split_training_data(self,training_data):
        self.training_data     = training_data
        print('DATA SET PREPARATON FOR TRAINING, TESTING AND VALIDATION\n')
        number_of_data              = int(self.training_data.describe()[self.training_data.columns[0]]['count'])
        print('\nnumber of data     =',number_of_data)
        
        ########## PREPARE FEATURE INPUTS ###############
        self.dict_x_origin               = {}
        
        for key in self.features_input:
            self.dict_x_origin[key] = self.training_data[key].values
            
        print('input_features     =', self.features_input)

        ########## PREPARE TARGET OUTPUTS ###############
        self.dict_y_origin = {}
        for key in self.target_output:
            self.dict_y_origin[key] =  self.training_data[key].values

        print('\ntarget_output      =', self.target_output)
        
        ########### SPLIT THE WHOLE DATA TO TRAIN - TEST - VALIDATION SETS ###############
        ####### 1st Seperate the Validation data from the whole data set ######## 
        indices1          = np.arange(number_of_data)
        split1            = train_test_split(indices1, test_size = 0.10, random_state=1038, shuffle=self.shuffle)
        indices1_train_test, indices1_valid = split1
        self.dict_x_train_test = {}
        self.dict_x_valid      = {}
        self.dict_y_train_test = {}
        self.dict_y_valid      = {}
        print('\nSPLITTING THE WHOLE DATA TO TRAIN-TEST AND VALIDATION SETS')
        for key in self.features_input:
            self.dict_x_train_test[key] = self.dict_x_origin[key][indices1_train_test].reshape((-1,1))
            self.dict_x_valid[key]      = self.dict_x_origin[key][indices1_valid].reshape((-1,1))
        for key in self.target_output:
            self.dict_y_train_test[key] = self.dict_y_origin[key][indices1_train_test].reshape((-1,1))
            self.dict_y_valid[key]      = self.dict_y_origin[key][indices1_valid].reshape((-1,1))
        print('\nDone!')
        #########################################################################
        ####### 2nd Split the train_test data sets into train and test ##########
        split2       = train_test_split(indices1_train_test, test_size = 0.10, random_state=2761, shuffle=True)
        indices2_train, indices2_test = split2
        self.dict_x_train = {}
        self.dict_x_test  = {}
        self.dict_y_train = {}
        self.dict_y_test  = {}
        print('\nSPLITTING THE TRAIN-TEST DATA TO TRAIN AND TEST SETS')
        for key in self.features_input:
            self.dict_x_train[key] = self.dict_x_origin[key][indices2_train].reshape((-1,1))
            self.dict_x_test[key]  = self.dict_x_origin[key][indices2_test].reshape((-1,1))
        for key in self.target_output:
            self.dict_y_train[key] = self.dict_y_origin[key][indices2_train].reshape((-1,1))
            self.dict_y_test[key]  = self.dict_y_origin[key][indices2_test].reshape((-1,1))
        print('\nDone')
        print('\nPRINTING TRAIN, TEST AND VALIDATION SETS')
        print('\n ***Input Features***\n') 
        for key in self.features_input:
            print('-----------------------------------------------')
            print('input features are       : ',key)
            print('train set shape for',key,'is          :' ,self.dict_x_train[key].shape)
            print('test set shape for',key,'is           :' ,self.dict_x_test[key].shape)
            print('validation set shape for',key,'is     :' ,self.dict_x_valid[key].shape)
        print('\n ***Output Targets***')    
        for key in self.target_output:
            print('-----------------------------------------------')
            print(key,'label shape for train_set is      :', self.dict_y_train[key].shape)
            print(key,'label shape for test_set is       :', self.dict_y_test[key].shape)
            print(key,'label shape for validation_set is :', self.dict_y_valid[key].shape)
        print('\n****************************************************************************\n')

        self.dict_x = {}
        self.dict_y = {}
        for key in ['train','test','valid']:
            self.dict_x[key] = {}
            self.dict_y[key] = {}
        self.dict_x['train'] = self.dict_x_train
        self.dict_x['test']  = self.dict_x_test
        self.dict_x['valid'] = self.dict_x_valid
        self.dict_y['train'] = self.dict_y_train
        self.dict_y['test']  = self.dict_y_test
        self.dict_y['valid'] = self.dict_y_valid

        ###################### SCALING input ###############################
        self.dict_scalerx    = {}
        self.dict_x_train_sc = {}
        self.dict_x_test_sc  = {}
        self.dict_x_valid_sc = {}
        for key in self.features_input:
            if self.scaler == 'minmax':
                scalerx = MinMaxScaler((-0.5,0.5))
            elif self.scaler == 'robust':
                scalerx = RobustScaler()
            elif self.scaler == 'standard':
                scalerx = StandardScaler()

            scx                       = scalerx.fit(self.dict_x_train[key])
            self.dict_x_train_sc[key] = scx.transform(self.dict_x_train[key])
            self.dict_x_test_sc[key]  = scx.transform(self.dict_x_test[key])
            self.dict_x_valid_sc[key] = scx.transform(self.dict_x_valid[key])
            self.dict_scalerx[key]    = scx
        ###################### SCALING output ###############################
        self.dict_scalery    = {}
        self.dict_y_train_sc = {}
        self.dict_y_test_sc  = {}
        self.dict_y_valid_sc = {}
        for key in self.target_output:
            if self.scaler == 'minmax':
                scalery = MinMaxScaler((-0.5,0.5))
            elif self.scaler == 'robust':
                scalery = RobustScaler()
            elif self.scaler == 'standard':
                scalery = StandardScaler()
            
            scy                       = scalery.fit(self.dict_y_train[key])
            self.dict_y_train_sc[key] = scy.transform(self.dict_y_train[key])
            self.dict_y_test_sc[key]  = scy.transform(self.dict_y_test[key])
            self.dict_y_valid_sc[key] = scy.transform(self.dict_y_valid[key])
            self.dict_scalery[key]    = scy

        self.dict_x_sc = {}
        self.dict_y_sc = {}
        for key in ['train','test','valid']:
            self.dict_x_sc[key] = {}
            self.dict_x_sc[key] = {}
        self.dict_x_sc['train'] = self.dict_x_train_sc
        self.dict_x_sc['test']  = self.dict_x_test_sc
        self.dict_x_sc['valid'] = self.dict_x_valid_sc
        self.dict_y_sc['train'] = self.dict_y_train_sc
        self.dict_y_sc['test']  = self.dict_y_test_sc
        self.dict_y_sc['valid'] = self.dict_y_valid_sc
    
    def split_test_data(self,test_data):
        
        self.test_data = test_data
        self.namekey.append('flight'+str(self.counter+1))
        print('\n Test data will be named as %s data from this point\n' % (self.namekey[self.counter]))
        
        print('DATA SET PREPARATON FOR FLIGHT DATA\n')
        number_of_data              = int(self.test_data.describe()[self.test_data.columns[0]]['count'])
        print('\nnumber of data     =',number_of_data)
        
        #######################################################################################        
        ########## PREPARE FEATURE INPUTS ###############
        self.dict_x_flight_origin               = {}
        self.dict_x_flight_origin[self.namekey[self.counter]] = {key: self.test_data[key].values for key in self.features_input}
            
        print('input_features     =',self.features_input)
        print('\n')
        
        #######################################################################################        
        ########## PREPARE TARGET OUTPUTS ###############
        self.dict_y_flight_origin = {}
        self.dict_y_flight_origin[self.namekey[self.counter]] =  {key: self.test_data[key].values for key in self.target_output}

        print('target_output      =',self.target_output)
        print('\n')
        
        ########### SPLIT THE WHOLE DATA TO TRAIN - TEST - VALIDATION SETS ###############
        ####### 1st Seperate the Validation data from the whole data set ######## 
        self.dict_x_flight     = {}
        self.dict_y_flight     = {}
        self.dict_x_flight[self.namekey[self.counter]] = {key :self.dict_x_flight_origin[self.namekey[self.counter]][key].reshape((-1,1)) for key in self.features_input}
        self.dict_y_flight[self.namekey[self.counter]] = {key :self.dict_y_flight_origin[self.namekey[self.counter]][key].reshape((-1,1)) for key in self.target_output}
        print('Done!')
        print('\n')
        ###################### SCALING ###############################
        self.dict_x_flight_sc = {}
        self.dict_y_flight_sc = {}
        self.dict_x_flight_sc[self.namekey[self.counter]] = {key :self.dict_scalerx[key].transform(self.dict_x_flight[self.namekey[self.counter]][key]) for key in self.features_input}
        self.dict_y_flight_sc[self.namekey[self.counter]] = {key :self.dict_scalery[key].transform(self.dict_y_flight[self.namekey[self.counter]][key]) for key in self.target_output}

        self.dict_x[self.namekey[self.counter]]    = self.dict_x_flight[self.namekey[self.counter]]
        self.dict_x_sc[self.namekey[self.counter]] = self.dict_x_flight_sc[self.namekey[self.counter]]
        self.dict_y[self.namekey[self.counter]]    = self.dict_y_flight[self.namekey[self.counter]]
        self.dict_y_sc[self.namekey[self.counter]] = self.dict_y_flight_sc[self.namekey[self.counter]]
        
        self.counter = self.counter + 1
        
