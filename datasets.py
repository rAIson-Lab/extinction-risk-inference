from foldrm import Classifier
import numpy as np

def extinction_birds(data_path='datasets/Extinction/AvoIUCNbehavMig.csv'):
    attrs = ['IslandEndemic','Mass','HWI','Habitat.x','Beak.Length.culmen','Beak.Length.nares',
            'Beak.Width','Beak.Depth','Tarsus.Length','Wing.Length','Kipps.Distance','Secondary1',
            'Tail.Length','LogRangeSize','Diet','Foraging','Migration','MatingSystem','NestPlacement','Territoriality',
            'IslandDwelling','LogClutchSize','LogNightLights','LogHumanPopulationDensity',
            'Extinct_full','Extinct_partial','Marine_full','Marine_partial','Migr_dir_full','Migr_dir_partial',
            'Migr_dir_local','Migr_disp_full','Migr_disp_partial','Migr_disp_local','Migr_altitudinal',
            'Irruptive','Nomad_full','Nomad_partial','Nomad_local','Resid_full','Resid_partial',
            'Unknown','Uncertain','Migratory_status','Migratory_status_2','Migratory_status_3']
    
    nums = ['Mass', 'HWI','Beak.Length.culmen','Beak.Length.nares','Beak.Width','Beak.Depth','Tarsus.Length',
            'Wing.Length','Kipps.Distance','Secondary1','Tail.Length','RedlistCategory','LogRangeSize',
            'LogBodyMass','LogClutchSize','LogNightLights','LogHumanPopulationDensity']
    label = 'Threat'
    model = Classifier(attrs=attrs, numeric=nums, label=label)
    data = model.load_data(data_path) # Use the argument here
    print('\n% extinction birds dataset loaded', np.shape(data))
    return model, data

def updated_extinction_birds(data_path='datasets/Extinction/Avo_Birdbase.csv'):
    attrs = ["Beak.Length_Culmen","Beak.Length_Nares","Beak.Width","Beak.Depth",
             "Tarsus.Length","Wing.Length","Kipps.Distance","Secondary1","Hand.Wing.Index",
             "Tail.Length","Habitat.Density","Migration","Primary.Lifestyle","Min.Latitude",
             "Max.Latitude","Centroid.Latitude","Centroid.Longitude","Range.Size","IUCN2024",
             "RR","ISL","RLM","LAT","Elevational.Range","HB","DB",
             "Nest_Type","Nest_SBS","Clutch_Max","Flightlessness","BodyMass","Order2",
             "Family","Habitat1"]
    
    nums = ["Beak.Length_Culmen","Beak.Length_Nares","Beak.Width","Beak.Depth",
             "Tarsus.Length","Wing.Length","Kipps.Distance","Secondary1","Hand.Wing.Index",
             "Tail.Length","Min.Latitude","Max.Latitude","Centroid.Latitude","Centroid.Longitude",
             "Range.Size","Elevational.Range","HB","DB","Clutch_Max","BodyMass"]
    label = "extinction_risk"
    model = Classifier(attrs=attrs, numeric=nums, label=label)
    data = model.load_data(data_path) # Use the argument here
    # Filtrar solo filas con extinction_risk igual a '0' o '1'
    label_index = attrs.index(label) if label in attrs else -1
    filtered_data = [row for row in data if str(row[label_index]) in ['0', '1']]
    print(f"\n% extinction birds dataset loaded (filtrado 0/1): {np.shape(filtered_data)}")
    return model, filtered_data

def extinction_amphib(data_path='datasets/Extinction/Pottieretal2022_ampthermtoler1.csv'):
    attrs = ["order","family","latitude","longitude","elevation",
             "acclimation_temp","acclimation_time",
             "medium_test_temp","start_temp","ramping","mean_UTL"
]
    nums = ["latitude","longitude","elevation",
             "acclimation_temp","acclimation_time",
             "medium_test_temp","start_temp","ramping","mean_UTL"]

    model = Classifier(attrs=attrs, numeric=nums, label="IUCN_status")
    data = model.load_data(data_path)
    print('\n% dataset', np.shape(data))
    return model, data
