import sys, os
sys.path.append(os.path.join(os.getcwd(), 'CONFOLD'))

import numpy as np
import pandas as pd
from CONFOLD.foldrm import Classifier

class MyClassifier(Classifier):
        def load_data(self, file, amount=-1):
            data, self.attrs = use_dataframe(file, self.attrs, self.label, self.numeric, amount)
            return data
        

def use_dataframe(file, attrs, label, numeric, amount):
    df = pd.read_csv(file, sep=',', on_bad_lines='skip') #get the dataframe

    #split into x and y
    df_x = df[attrs]
    df_y = df[label]
    result = pd.concat([df_x,df_y], axis=1)
    attrs.append(label)
    return result,attrs
        
def final_extinctionrisk(data_path='datasets/Extinction/traits_combined_noNA_5Dece25.csv'):
    attrs = ["Order","Family","Agriculture","Hunting","Invasive_species","Climate_change",
             "Beak_length_culmen","Beak_depth",
             "Tarsus_length","Wing_length","Hand_wing_index","Tail_length","Minimum_latitude","Maximum_latitude",
             "Primary_lifestyle","Island_restricted_breeding","Latitudinal_range","Elevational_range","Habitat_breadth",
             "Diet_breadth","Realm","Minimum_elevation","Maximum_elevation","Adult_survival_annual","Generation_length",
             "Range_size","Body_mass","Clutch_size","Diet","Habitat","Migration","Extinction_risk"]
    
    nums = ["Beak_length_culmen","Beak_depth","Tarsus_length","Wing_length","Hand_wing_index","Tail_length",
            "Minimum_latitude","Maximum_latitude","Minimum_elevation","Elevational_range","Maximum_elevation",
            "Habitat_breadth","Diet_breadth","Adult_survival_annual","Generation_length","Range_size","Body_mass",
            "Clutch_size"]
    label = "Extinction_risk"
    
    model = Classifier(attrs=attrs, numeric=nums, label=label)
    data = model.load_data(data_path)
    return model, data

def final_extinctionrisk_noth(data_path='datasets/Extinction/traits_combined_noNA_5Dece25.csv'):
    attrs = ["Order","Family",
             "Beak_length_culmen","Beak_depth",
             "Tarsus_length","Wing_length","Hand_wing_index","Tail_length","Minimum_latitude","Maximum_latitude",
             "Primary_lifestyle","Island_restricted_breeding","Latitudinal_range","Elevational_range","Habitat_breadth",
             "Diet_breadth","Realm","Minimum_elevation","Maximum_elevation","Adult_survival_annual","Generation_length",
             "Range_size","Body_mass","Clutch_size","Diet","Habitat","Migration","Extinction_risk"]
    
    nums = ["Beak_length_culmen","Beak_depth","Tarsus_length","Wing_length","Hand_wing_index","Tail_length",
            "Minimum_latitude","Maximum_latitude","Minimum_elevation","Elevational_range","Maximum_elevation",
            "Habitat_breadth","Diet_breadth","Adult_survival_annual","Generation_length","Range_size","Body_mass",
            "Clutch_size"]
    label = "Extinction_risk"
    
    model = Classifier(attrs=attrs, numeric=nums, label=label)
    data = model.load_data(data_path)
    return model, data

def final_extinctionrisk_dataframe(data_path='datasets/Extinction/traits_combined_noNA_5Dece25.csv'):
    attrs = ["Order","Family","Agriculture","Hunting","Invasive_species","Climate_change",
             "Beak_length_culmen","Beak_depth",
             "Tarsus_length","Wing_length","Hand_wing_index","Tail_length","Minimum_latitude","Maximum_latitude",
             "Primary_lifestyle","Island_restricted_breeding","Latitudinal_range","Elevational_range","Habitat_breadth",
             "Diet_breadth","Realm","Minimum_elevation","Maximum_elevation","Adult_survival_annual","Generation_length",
             "Range_size","Body_mass","Clutch_size","Diet","Habitat","Migration"]
    
    nums = ["Beak_length_culmen","Beak_depth","Tarsus_length","Wing_length","Hand_wing_index","Tail_length",
            "Minimum_latitude","Maximum_latitude","Minimum_elevation","Elevational_range","Maximum_elevation",
            "Habitat_breadth","Diet_breadth","Adult_survival_annual","Generation_length","Range_size","Body_mass",
            "Clutch_size"]
    label = "Extinction_risk"
    
    model = MyClassifier(attrs=attrs, numeric=nums, label=label)
    data = model.load_data(data_path)
    return model, data

def final_extinctionrisk_noth_dataframe(data_path='datasets/Extinction/traits_combined_noNA_5Dece25.csv'):
    attrs = ["Order","Family",
             "Beak_length_culmen","Beak_depth",
             "Tarsus_length","Wing_length","Hand_wing_index","Tail_length","Minimum_latitude","Maximum_latitude",
             "Primary_lifestyle","Island_restricted_breeding","Latitudinal_range","Elevational_range","Habitat_breadth",
             "Diet_breadth","Realm","Minimum_elevation","Maximum_elevation","Adult_survival_annual","Generation_length",
             "Range_size","Body_mass","Clutch_size","Diet","Habitat","Migration"]
    
    nums = ["Beak_length_culmen","Beak_depth","Tarsus_length","Wing_length","Hand_wing_index","Tail_length",
            "Minimum_latitude","Maximum_latitude","Minimum_elevation","Elevational_range","Maximum_elevation",
            "Habitat_breadth","Diet_breadth","Adult_survival_annual","Generation_length","Range_size","Body_mass",
            "Clutch_size"]
    label = "Extinction_risk"
    
    model = MyClassifier(attrs=attrs, numeric=nums, label=label)
    data = model.load_data(data_path)
    return model, data
