# Create Files for Simulink Export
from import_modules import *

class export_wghts_and_biases:
    def __init__(self,models,filename_dataset,filename_wghts='all_models.mat',
                 feature_keys = feature_keys, target_keys= target_keys):
        
        self.models           = models
        self.filename_dataset = filename_dataset
        self.filename_wghts   = filename_wghts
        self.feature_keys     = feature_keys
        self.target_keys      = target_keys
        ####################### if 'Python2Simulink' is not in the directory, copy it #####################
        if 'Python2Simulink' not in os.listdir():
            shutil.copytree('/media/veracrypt1/DigitalTwin/Python2Simulink',current_dir+'Python2Simulink')
            print('Python2Simulink folder copied to destination\n')
        else:
            print('Python2Simulink exists in the destination\n')
        ################################################################################################### 
        
        ####################### weights and biases exported to target directory ###########################
        wghts  = {}
        layers = {}
        for key in list(self.models.keys()):
            wghts[key] = self.models[key].model.get_weights()
            for ii in range(int(len(self.models[key].model.get_weights())/2)):
                layers[key+'_layer'+str(ii+1)+'_weights'] = wghts[key][2*ii].transpose()
                layers[key+'_layer'+str(ii+1)+'_bias']    = wghts[key][2*ii+1].reshape(-1,1)
            print('%s weights has been exported\n' % (key))
            
        sio.savemat(current_dir+'Python2Simulink/'+self.filename_weights,layers)
        ####################################################################################################
        
        ############################ Create Files for Matlab to Read and Run ###############################
        for key in list(self.models.keys()):
            self.models[key].scaler_path['feature'] = current_dir+'MatFiles/'+self.models[key].model_name+'_featurescaler.mat'
            self.models[key].scaler_path['target']  = current_dir+'MatFiles/'+self.models[key].model_name+'_targetscaler.mat'
            self.models[key].writeStandartScaler_AsMatFile(self.models[key].dict_scalerx,self.models[key].scaler_path['feature'],self.feature_keys)
            self.models[key].writeStandartScaler_AsMatFile(self.models[key].dict_scalery,self.models[key].scaler_path['target'],self.target_keys)

        Input_Output      = [str(len(self.feature_keys)),str(len(self.target_keys)),str(len(self.models.keys()))]
        Input_Output_file = open(current_dir+'Python2Simulink/input_output_for_matlab.txt','w+')
        mat_file          = open(current_dir+'Python2Simulink/mat_files_for_matlab.txt','w+')
        mat_file_paths    = []
        
        mat_file_paths.append(os.getcwd()+'/'+self.filename)
        mat_file_paths.append(filename_weights)
        for key in list(self.models.keys()):
            mat_file_paths.append(self.models[key].scaler_path['feature'])
            mat_file_paths.append(self.models[key].scaler_path['target'])
            mat_file_paths.append(self.models[key].model_path)
            Input_Output.append(str(self.models[key].total_layer_no))

        for i in range(len(Input_Output)):
            Input_Output_file.write(Input_Output[i])
            Input_Output_file.write('\n')
            
        for i in range(len(mat_file_paths)):
            mat_file.write(mat_file_paths[i])
            mat_file.write('\n')
        
        Input_Output_file.close()
        mat_file.close()
        
        print('input out file created for matlab\n')
        print('file consisting mat file paths created for matlab\n')
        #####################################################################################################