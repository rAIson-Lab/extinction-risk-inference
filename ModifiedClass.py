import sys, os
sys.path.append(os.path.join(os.getcwd(), 'CONFOLD'))

import numpy as np
import pandas as pd
from CONFOLD.foldrm import Classifier

class MyClassifier(Classifier):
        def load_data(self, file, amount=-1):
            data, self.attrs = use_dataframe(file, self.attrs, self.label, self.numeric, amount)
            return data
        
def extinction_birds2(data_path='datasets/Extinction/AvoIUCNbehavMig.csv'):
    attrs = ['IslandEndemic','Mass','HWI','Habitat.x','Beak.Length.culmen','Beak.Length.nares',
            'Beak.Width','Beak.Depth','Tarsus.Length','Wing.Length','Kipps.Distance','Secondary1',
            'Tail.Length','LogRangeSize','Diet','Foraging','Migration','MatingSystem','NestPlacement','Territoriality',
            'IslandDwelling','LogClutchSize','LogNightLights','LogHumanPopulationDensity',
            'Extinct_full','Extinct_partial','Marine_full','Marine_partial','Migr_dir_full','Migr_dir_partial',
            'Migr_dir_local','Migr_disp_full','Migr_disp_partial','Migr_disp_local','Migr_altitudinal',
            'Irruptive','Nomad_full','Nomad_partial','Nomad_local','Resid_full','Resid_partial',
            'Unknown','Uncertain','Migratory_status','Migratory_status_2','Migratory_status_3', 'LogBodyMass', 'RedlistCategory']
    nums = ['Mass', 'HWI','Beak.Length.culmen','Beak.Length.nares','Beak.Width','Beak.Depth','Tarsus.Length',
            'Wing.Length','Kipps.Distance','Secondary1','Tail.Length','LogRangeSize',
            'LogClutchSize','LogNightLights','LogHumanPopulationDensity']
    
    label = 'Threat'
    #model = MyClassifier(attrs=attrs, numeric=nums, label=label)
    model = MyClassifier(attrs=attrs, numeric=nums, label=label)
    data = model.load_data(data_path) # Use the argument here
    return model, data

def use_dataframe(file, attrs, label, numeric, amount):
    df = pd.read_csv(file) #get the dataframe

    #split into x and y
    df_x = df[attrs]
    df_y = pd.to_numeric(df[label])
    result = pd.concat([df_x,df_y], axis=1)
    attrs.append(label)
    return result,attrs