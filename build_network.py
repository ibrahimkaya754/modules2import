from import_modules import *
from helper_functions import *

class prepare_inputs:
    def __init__(self, data_splitted, list_tests, list_params,
                 feature_keys, target_keys, cnn=False):
        
        ##########################################################################################################
        self.data_splitted              = data_splitted
        self.list_tests                 = list_tests
        self.list_params                = list_params
        self.feature_keys               = feature_keys
        self.target_keys                = target_keys
        self.dict_x                     = {}
        self.dict_y                     = {}
        self.dict_x_sc                  = {}
        self.dict_y_sc                  = {}
        self.cnn                        = cnn

        self.list_split_through = [key for key in self.data_splitted.dict_x.keys()]
        for key in self.list_split_through:
            self.dict_x[key]    = self.data_splitted.dict_x[key]
            self.dict_y[key]    = self.data_splitted.dict_y[key]
            self.dict_x_sc[key] = self.data_splitted.dict_x_sc[key]
            self.dict_y_sc[key] = self.data_splitted.dict_y_sc[key]

        print('Shapes of the %s and  Sets of Input Features \n' % (self.list_split_through))
        for data in self.list_split_through:
            print('--------------------------------------------------------------')
            for feature in self.feature_keys:
                print(data,' input set shape for ',feature,' is: ', self.dict_x_sc[data][feature].shape)
            
        print('\n*******************************************************************************\n')
        # Shapes of the Train, Test and Validation Sets of Output Targets
        print('Shapes of the %s and  Sets of Output Targets \n' % (self.list_split_through))
        for data in self.list_split_through:
            print('--------------------------------------------------------------')
            for target in self.target_keys:
                print(data,' target output set shape for ',target,' is: ', self.dict_y_sc[data][target].shape)

        print('\n')
        ##########################################################################################################
        self.input_dl      = {}
        self.output_dl     = {}

        for key in self.list_split_through:
            self.input_dl[key]  = {'input_all'   : self.dict_x_sc[key][self.feature_keys[0]]}
            self.output_dl[key] = {'all_targets' : self.dict_y_sc[key][self.target_keys[0]]}

        for key in self.list_split_through:
            for param in self.feature_keys[1:]:
                self.input_dl[key]['input_all']    =  np.hstack((self.input_dl[key]['input_all'],self.dict_x_sc[key][param]))
            for param in self.target_keys[1:]:
                self.output_dl[key]['all_targets'] =  np.hstack((self.output_dl[key]['all_targets'],self.dict_y_sc[key][param]))

        if self.cnn:
            self.input_all                      = Input(shape=(self.input_dl['train']['input_all'].shape[1],1), name='input_all')
            for key in self.list_split_through:
                self.input_dl[key]['input_all'] = self.input_dl[key]['input_all'].reshape(self.input_dl[key]['input_all'].shape[0],
                                                                                          self.input_dl[key]['input_all'].shape[1],1)
        else:
            self.input_all = Input(shape=(self.input_dl['train']['input_all'].shape[1],), name='input_all')
        ##########################################################################################################

# BUILD CUSTOM NETWORK
class model(prepare_inputs):
    def __init__(self, data_splitted, list_tests, list_params,
                 feature_keys, target_keys, cnn, mdl_name, act='tanh', 
                 trainable_layer=True, initializer='glorot_normal', applyact2lastlayer=False):

        super().__init__(data_splitted, list_tests, list_params,
                 feature_keys, target_keys,cnn=False)

        self.model_name      = mdl_name
        self.act             = act
        self.trainable_layer = trainable_layer
        self.init            = initializer
        self.opt             = Adam(lr=0.001)
        self.losses          = {}
        self.lossWeights     = {}
        self.scaler_path     = {'feature' : None,
                                'target'  : None}
        self.regularization_paramater = 0.0
        self.dict_scalery    = data_splitted.dict_scalery
        self.dict_scalerx    = data_splitted.dict_scalerx
        self.applyact2lastlayer = applyact2lastlayer

    def autoencoder(self,list_nn=[150,100,20],bottleneck=3,load_weights=False):
        self.list_nn      = list_nn
        self.bottleneck   = bottleneck
        self.load_weights = load_weights

        self.L1 = Dense(self.list_nn[0], activation=self.act,
                     kernel_initializer=self.init, trainable = self.trainable_layer,
                     kernel_regularizer=regularizers.l2(self.regularization_paramater))(self.input_all)

        for ii in range(1,len(self.list_nn)):
            self.L1 = Dense(self.list_nn[ii], activation=self.act, trainable = self.trainable_layer,
                         kernel_initializer=self.init,
                         kernel_regularizer=regularizers.l2(self.regularization_paramater))(self.L1)    

        self.L1 = Dense(self.bottleneck, activation='linear', name='bottleneck',
                         kernel_initializer=self.init,
                         kernel_regularizer=regularizers.l2(self.regularization_paramater))(self.L1)
        
        for ii in range(0,len(self.list_nn)):
            self.L1 = Dense(self.list_nn[-ii-1], activation=self.act, trainable = self.trainable_layer,
                         kernel_initializer=self.init,
                         kernel_regularizer=regularizers.l2(self.regularization_paramater))(self.L1)

        self.LOut = Dense(len(self.target_keys), activation=self.act, name='all_targets',
                         kernel_initializer=self.init,
                         kernel_regularizer=regularizers.l2(self.regularization_paramater))(self.L1)
                        
        self.autoencoder_is_exist = True
        self.__build_model__()
    
    def neuralnet(self,list_nn=[250,200,150,50,10],load_weights=False):
        self.list_nn         = list_nn
        self.load_weights    = load_weights
        self.total_layer_no  = len(self.list_nn)+1

        self.L1 = Dense(self.list_nn[0], activation=self.act,
                     kernel_initializer=self.init, trainable = self.trainable_layer,
                     kernel_regularizer=regularizers.l2(self.regularization_paramater))(self.input_all)

        for ii in range(1,len(self.list_nn)):
            self.L1 = Dense(self.list_nn[ii], activation=self.act, trainable = self.trainable_layer,
                         kernel_initializer=self.init,
                         kernel_regularizer=regularizers.l2(self.regularization_paramater))(self.L1)    

        if self.applyact2lastlayer:
            act = self.act
        else:
            act = 'linear'
        self.LOut = Dense(len(self.target_keys), activation=act, name='all_targets',
                         kernel_initializer=self.init,
                         kernel_regularizer=regularizers.l2(self.regularization_paramater))(self.L1)

        self.autoencoder_is_exist = False
        self.__build_model__()
                        
    def __build_model__(self):
        self.model                         = Model(inputs=[self.input_all], outputs=self.LOut)
        self.description                   = None
        self.losses['all_targets']         = mean_squared_error
        self.lossWeights['all_targets']    = 1.0
        self.model_path                    = os.getcwd()+"/" + self.model_name + '.hdf5'
        self.learning_rate_decrease_factor = 0.97
        self.learning_rate_patience        = 5
        self.number_of_params              = self.model.count_params()
        self.reduce_lr                     = ReduceLROnPlateau(monitor='val_loss', 
                                                               factor=self.learning_rate_decrease_factor,
                                                               patience=self.learning_rate_patience, 
                                                               min_lr=0.0000001, mode='min', verbose=1)
        self.checkpoint                    = ModelCheckpoint(self.model_path, 
                                                             monitor='val_loss', verbose=1, 
                                                             save_best_only=True, period=1, 
                                                             mode='min',save_weights_only=False)
        self.model.compile(optimizer=self.opt, loss=self.losses['all_targets'], metrics=['mse'])
        plot_model(self.model,to_file=self.model_name+'.png', show_layer_names=True,show_shapes=True)
        print('\n%s with %s params created' % (self.model_name,self.number_of_params))
        if os.path.exists(self.model_path):
            if self.load_weights:
                print('weights loaded for %s' % (self.model_name))
                self.model.load_weights(self.model_path)
 
        # Make the prediction for bottleneck layer
        if self.autoencoder_is_exist:
            self.bottleneck_layer = Model(self.model.input,self.model.get_layer('bottleneck').output)
            self.target_bn = ['dim'+str(ii) for ii in range(self.bottleneck)]
        
    def __describe__(self):
        return self.description
     
    def summary(self):
        self.model.summary()
        print('\nModel Name is: ',self.model_name)
        print('\nModel Path is: ',self.model_path)
        print('\nActivation Function is: ',self.act)
        print('\nLearning Rate Decreases by a factor of %s with patience of %s' % (self.learning_rate_decrease_factor,
                                                                                   self.learning_rate_patience))
        if self.description != None:
            print('\nModel Description: '+self.__describe__())
        
    def run(self,num_epochs,batch_size):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        print('Start Running \n')
        self.history = self.model.fit(self.input_dl['train'],
                                      self.output_dl['train'], 
                                      batch_size=self.batch_size, epochs=self.num_epochs, shuffle=True,
                                      callbacks=[self.checkpoint, self.reduce_lr],
                                      validation_data=(self.input_dl['test'],self.output_dl['test']), verbose=1)
        self.val_loss = np.min(self.history.history['val_loss'])
        
    def results(self,load_weights=False):
        if load_weights:
            self.model.load_weights(self.model_path)
            print('Weights Loaded')
        
        self.out_dl_predicted            = {}
        for data in self.list_split_through:
            self.out_dl_predicted[data]            = {'all_target' : self.model.predict(self.input_dl[data], batch_size=None)}
            print('Prediction for %s set is completed' % (data))
            for value,key in enumerate(self.target_keys):
                self.out_dl_predicted[data][key]            = {'scaled'  : self.out_dl_predicted[data]['all_target'][:,value]}
                self.out_dl_predicted[data][key]['inverse'] = self.dict_scalery[key].inverse_transform(self.out_dl_predicted[data][key]['scaled'].reshape(-1,1))
        print('-------------------------------------------------------------------------------------\n')
        for data in self.list_split_through:
            print('\nExplained Variance Calculation for %s set' % (data))
            for param in self.target_keys:
                print("Explained Variance of %s set for %s : %.8f" % (data,param,explained_variance_score(self.dict_y_sc[data][param],
                      self.out_dl_predicted[data][param]['scaled'].reshape(self.out_dl_predicted[data][param]['scaled'].shape[0],1))))
            print('-------------------------------------------------------------------------------------')

        if self.autoencoder_is_exist:
            self.out_dl_predicted_bottleneck = {}
            for data in self.list_split_through:
                self.out_dl_predicted_bottleneck[data] = {'all_target' : self.bottleneck_layer.predict(self.input_dl[data], batch_size=None)}
                print('Bottleneck Prediction for %s set is completed' % (data))
                for value,key in enumerate(self.target_bn):
                    self.out_dl_predicted_bottleneck[data][key] = {'scaled'  : self.out_dl_predicted_bottleneck[data]['all_target'][:,value]}

    def plots(self,pnt_number=250,plot_train=False,plot_test=False,plot_valid=False,plot_flight=False,inverseplot=False):
        self.pnt_number  = pnt_number
        self.inverseplot = inverseplot
        self.plot_list   = {namekey  :plot_flight for namekey in self.data_splitted.namekey}
        self.plot_list['train'] = plot_train
        self.plot_list['test']  = plot_test
        self.plot_list['valid'] = plot_valid
        for data in self.list_split_through:
            if 'flight' in data:
                self.pnt_number = -1
            if self.plot_list[data]:
                print('\nPlot for %s set\n' % (data))
                for param in self.target_keys:
                    print(param)
                    plt.figure(figsize=(26,9))
                    if self.inverseplot:
                        plt.plot(self.out_dl_predicted[data][param]['inverse'][0:self.pnt_number],'--',markersize=1,label='predicted',color='tab:red')
                        plt.plot(self.dict_y[data][param][0:self.pnt_number],'--', markersize=1, label='actual', color = 'tab:blue')
                    else:
                        plt.plot(self.out_dl_predicted[data][param]['scaled'][0:self.pnt_number],'--',markersize=1,label='predicted',color='tab:red')
                        plt.plot(self.dict_y_sc[data][param][0:self.pnt_number],'--', markersize=1, label='actual', color = 'tab:blue')
                    plt.legend()
                    plt.xlabel('sample point')
                    plt.ylabel(param)
                    plt.title('explained variance score for the %s set for %s is: ' % (data,param))
                    plt.grid()
                    plt.show()

    def scatter_plot_for_bottleneck(self):
        if self.autoencoder_is_exist:
            for value1,key1 in enumerate(self.target_bn):
                if key1 == self.target_bn[-1]:
                    break
                else:
                    for key2 in self.target_bn[value1+1:]:
                        if key1 == key2:
                            continue
                        else:
                            fig = plt.figure(figsize=(26,9))
                            for data in self.list_split_through:
                                plt.scatter(self.out_dl_predicted_bottleneck[data][key1]['scaled'],self.out_dl_predicted_bottleneck[data][key2]['scaled'],label=data)
                                plt.legend()
                                plt.xlabel(key1)
                                plt.ylabel(key2)
                                plt.grid()
                            plt.show()
                            #fig.savefig('./images/scatterplot_bottleneck_')
        else:
            print('bottleneck is applicable only for autoencoder')
            
    def __mae__(self):
        self.mae = {}
        for data in self.list_split_through:
            self.mae[data] = {param:np.zeros((self.out_dl_predicted[data][param]['scaled'].shape[0],1)) for param in self.target_keys}
        
        for data in self.list_split_through:
            for param in self.target_keys:
                self.mae[data][param] = self.out_dl_predicted[data][param]['scaled'].reshape(-1,1) - self.dict_y_sc[data][param]
          
    def histogram_mae(self):
        self.__mae__()
        for param in self.target_keys:
            print('************ Histogram Plot of Mae for %s set****************\n' % (self.list_split_through))
            for value,data in enumerate(self.list_split_through):
                fig = plt.figure(figsize=(25,36))
                plt.subplot(411+value)
                plt.hist(self.mae[data][param],label='%s for %s set' %(param,data),bins=500)
                plt.legend()
                plt.xlabel(param)
                plt.grid()
                plt.xlim((-0.50,+0.50))
            plt.show()
            #fig.savefig('./images/error_hist'+self.target_keys[ii])
            print("*******************************************************************************************************************")
            print("*******************************************************************************************************************")
                
    def corr(self):
        # Pearson Correlation for bottleneck dimensions
        if self.autoencoder_is_exist:
            self.covariance  = {}
            self.sigma       = {}
            self.correlation = {}

            for data in self.list_split_through:
                self.covariance[data]  = {}
                self.sigma[data]       = {}
                self.correlation[data] = {}
            
            for data in self.list_split_through:
                for dim in self.target_bn:
                    self.covariance[data][dim]  = {}
                    self.correlation[data][dim] = {}

            for data in self.list_split_through:
                for dim1 in self.target_bn:
                    self.sigma[data][dim1] = sigma(self.out_dl_predicted_bottleneck[data][dim1]['scaled'])
                    for dim2 in self.target_bn:
                        self.sigma[data][dim2] = sigma(self.out_dl_predicted_bottleneck[data][dim2]['scaled'])
                        self.covariance[data][dim1][dim2]  = covar(self.out_dl_predicted_bottleneck[data][dim1]['scaled'],self.out_dl_predicted_bottleneck[data][dim2]['scaled'])
                        self.correlation[data][dim1][dim2] = self.covariance[data][dim1][dim2] / (self.sigma[data][dim1] * self.sigma[data][dim2])
    
            # Scaler Plot for the Correlations of Dimensions Obtained in Bottleneck
            for data in self.list_split_through:
                print('\nCorrelation Coefficient for %s data' % (data))
                for dim1 in self.target_bn:
                    plt.figure(figsize=(26,9))
                    plt.scatter(np.arange(len(self.correlation[data][dim1])),[self.correlation[data][dim1][dim] for dim in self.target_bn], label= 'correlation for %s' % (dim1))
                    plt.legend()
                    plt.xlabel([dim for dim in self.target_bn])
                    plt.ylabel(dim1)
                    plt.title('Correlation for %s obtained from the prediction of bottleneck of AutoEncoder' % (dim1))
                    plt.grid()
                    plt.show()
        else:
            print('bottleneck correlation is applicable only for autoencoder')

    def mean_distance(self):
        if self.autoencoder_is_exist:
            print('Mean Distance for the bottleneck dimensions')
            self.shuffling = {}
            self.mean_dist = {}
            self.mean      = {}
            self.sigma     = {}

            for data in self.list_split_through:
                self.shuffling[data] = np.random.permutation(np.arange(self.out_dl_predicted_bottleneck[data][self.target_bn[0]]['scaled'].shape[0]))
                self.mean_dist[data] = {}
                self.mean[data]      = {}
                self.sigma[data]     = {}

            for data in self.list_split_through:
                for dim in self.target_bn:
                    self.out_dl_predicted_bottleneck[data][dim]['scaled_shuffled'] = self.out_dl_predicted_bottleneck[data][dim]['scaled'][self.shuffling[data]]
                for param in self.target_keys:
                    self.out_dl_predicted[data][param]['scaled_shuffled']          = self.out_dl_predicted[data][param]['scaled'][self.shuffling[data]]
                    self.out_dl_predicted[data][param]['inverse_shuffled']         = self.out_dl_predicted[data][param]['inverse'][self.shuffling[data]]
                    self.out_dl_predicted[data][param]['scaled_shuffled_outlier']  = []
                    self.out_dl_predicted[data][param]['inverse_shuffled_outlier'] = []
                    self.out_dl_predicted[data][param]['scaled_outlier']           = []

            for data in self.list_split_through:
                for dim in self.target_bn:
                    self.sigma[data][dim] = {'original' : np.std(self.out_dl_predicted_bottleneck[data][dim]['scaled_shuffled'])}
                    self.mean[data][dim]  = {'original' : np.mean(self.out_dl_predicted_bottleneck[data][dim]['scaled_shuffled'])}

            for dim in self.target_bn:
                for data1 in self.list_split_through:
                    self.out_dl_predicted_bottleneck[data1][dim]['ztransformed'] = {data2 : (self.out_dl_predicted_bottleneck[data1][dim]['scaled_shuffled'] - \
                                                                                            self.mean[data2][dim]['original'])/self.sigma[data2][dim]['original'] for data2 in self.list_split_through}

            for dim in self.target_bn:
                for data1 in self.list_split_through:
                    self.sigma[data1][dim]['ztransformed'] = {data2 : np.std(self.out_dl_predicted_bottleneck[data1][dim]['ztransformed'][data2]) for data2 in self.list_split_through}
                    self.mean[data1][dim]['ztransformed']  = {data2 : np.mean(self.out_dl_predicted_bottleneck[data1][dim]['ztransformed'][data2]) for data2 in self.list_split_through}
            
            # mean distance is calculated against the 'train' data of ztransformed predictions, every data (train, test, valid and flight) must be compared with the ztransformed train data
            # furthermore, the predicted train, test, valid and flight data must be ztransformed according to the base data which is the train data, here.
            # so we need to use --> out_dl_predicted_bottleneck[data][dim]['ztransformed']['train'] - mean['train'][dim]['ztransformed']['train']
            for data in self.list_split_through:
                for dim in self.target_bn:
                    self.mean_dist[data][dim] = {'all_data'        :np.abs(self.out_dl_predicted_bottleneck[data][dim]['ztransformed']['train'] - self.mean['train'][dim]['ztransformed']['train']),
                                                'outlier_indices' :np.where(np.abs(self.out_dl_predicted_bottleneck[data][dim]['ztransformed']['train'] - self.mean['train'][dim]['ztransformed']['train']) > \
                                                                    (self.mean['train'][dim]['ztransformed']['train']+3*self.sigma['train'][dim]['ztransformed']['train']))}

            for data in self.list_split_through:
                for dim in self.target_bn:
                    self.out_dl_predicted_bottleneck[data][dim]['scaled_shuffled_outlier'] = self.out_dl_predicted_bottleneck[data][dim]['scaled_shuffled'][self.mean_dist[data][dim]['outlier_indices'][0]]
                    self.out_dl_predicted_bottleneck[data][dim]['scaled_outlier']          = self.out_dl_predicted_bottleneck[data][dim]['scaled']\
                                                                                            [self.shuffling[data][self.mean_dist[data][dim]['outlier_indices'][0]]]

            for data in self.list_split_through:
                for param in self.target_keys:
                    for dim in self.target_bn:
                        for datum in self.out_dl_predicted[data][param]['scaled_shuffled'][self.mean_dist[data][dim]['outlier_indices'][0]]:
                            self.out_dl_predicted[data][param]['scaled_shuffled_outlier'].append(datum)
                        for datum in self.out_dl_predicted[data][param]['inverse_shuffled'][self.mean_dist[data][dim]['outlier_indices'][0]]:
                            self.out_dl_predicted[data][param]['inverse_shuffled_outlier'].append(datum)
                        for datum in self.out_dl_predicted[data][param]['scaled'][self.shuffling[data][self.mean_dist[data][dim]['outlier_indices'][0]]]:
                            self.out_dl_predicted[data][param]['scaled_outlier'].append(datum)

            # Scatter Plot for the Mean Distance
            for data in self.list_split_through:
                for dim in self.target_bn:
                    fig = plt.figure(figsize=(26,9))
                    plt.scatter(np.arange(self.out_dl_predicted_bottleneck[data][dim]['scaled'].shape[0]), 
                                self.mean_dist[data][dim]['all_data'], 
                                label='%s vs train for %s' % (data,dim))
                    plt.scatter(np.arange(self.out_dl_predicted_bottleneck[data][dim]['scaled'].shape[0]), 
                                np.ones((self.out_dl_predicted_bottleneck[data][dim]['scaled'].shape[0],1)) * \
                                (self.mean['train'][dim]['ztransformed']['train']+3*self.sigma['train'][dim]['ztransformed']['train']), 
                                label='3 sigma distance from the mean of %s of the train data' % (dim))
                    plt.legend()
                    plt.xlabel('data')
                    plt.ylabel('distance between ztransformed %s of %s data according to train data' % (dim,data))
                    plt.grid()
                    plt.show()
        else:
            print('mean distance for bottleneck dimensions is applicable only for autoencoder')

    def writeStandartScaler_AsMatFile(self,scaler,fileName,keys):
        if os.path.exists('./MatFiles/')==False:
            os.makedirs('./MatFiles/')
        self.mean      = {}
        self.variance  = {}
        self.scale     = {}
        self.scaler    = {}
        for key in keys:
            self.mean[key]      = scaler[key].mean_
            self.variance[key]  = scaler[key].var_
            self.scale[key]     = scaler[key].scale_
        self.scaler['mean']     = self.mean
        self.scaler['variance'] = self.variance
        self.scaler['scale']    = self.scale
        sio.savemat(fileName, self.scaler)
        return self.scaler
    
    def writeMinMaxScaler_AsMatFile(self,scaler,fileName,keys):
        if os.path.exists('./MatFiles/')==False:
            os.makedirs('./MatFiles/')
        self.min      = {}
        self.max      = {}
        self.scale    = {}
        self.data_min = {}
        self.data_max = {}
        self.scaler   = {}

        for key in keys:
            self.min[key], self.max[key] = scaler[key].feature_range
            self.scale[key]              = scaler[key].scale_
            self.data_min[key]           = scaler[key].data_min_
            self.data_max[key]           = scaler[key].data_max_
        self.scaler['min']      = self.min
        self.scaler['max']      = self.max
        self.scaler['scale']    = self.scale
        self.scaler['data_min'] = self.data_min
        self.scaler['data_max'] = self.data_max
        sio.savemat(fileName, self.scaler)
        return self.scaler