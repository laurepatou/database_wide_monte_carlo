from brightway2 import *
import numpy as np
import os
import multiprocessing as mp
from scipy.sparse.linalg import factorized, spsolve
from scipy import sparse
import time
import datetime
import pickle
import click
import scipy as sp
import h5py
from bw2calc.matrices import MatrixBuilder
from stats_arrays.random import MCRandomNumberGenerator
import random

##################
# HDF5 functions #
##################

# All those functions work based on LCA objects from Brightway

# function to rebuild csr matrix from hdf5 storage

def hdf5_to_csr_matrix(hdf5_file,group_full_path):
    
    # Access hdf5 group of the csr info
    group=hdf5_file[group_full_path]
    
    #Rebuild csr matrix
    csr=sp.sparse.csr_matrix((group['data'][:],group['indices'][:],group['indptr'][:]), group['shape'][:])
    
    return csr;

#function to create list to store in hfd5 for LCA object _dict: biosphere_dict, activity_dict, product_dict
def LCA_dict_to_hdf5(LCA_dict,hdf5_file,group_path):
    
    # Retrieve or create the groups and subgroups
    group=hdf5_file.require_group(group_path)
    
    ###### WARNING : Modify the builder because _dict items are like 
    #####('ecoinvent 3.3 cutoff', 'c533b046462b6c56a5636ca177347c48'): 35
    #### Use .decode('UTF-8') to convert keys_1_list items for bytes to str
    
    
    keys_0=np.string_([key[0] for key in LCA_dict.keys()][0])
    keys_1_list=np.string_([key[1] for key in LCA_dict.keys()])
    items_list=[item for item in LCA_dict.values()]
    
    # Create datasets containing values of csr matrix
    group.create_dataset('keys_1',data=keys_1_list,compression="gzip")
    group.create_dataset('values',data=items_list,compression="gzip")
    group.create_dataset('keys_0',data=keys_0)

    return;

#function to rebuild LCA object _dict: biosphere_dict, activity_dict, product_dict
def hdf5_to_LCA_dict(hdf5_file,group_path):
    
    # Retrieve or create the groups and subgroups
    group=hdf5_file.require_group(group_path)
    
    #Retrieve Keys and Items
    keys_0=group['keys_0'][()].decode('UTF-8')
    keys_1_list=[key_1.decode('UTF-8') for key_1 in group['keys_1'][()]] #### Use .decode('UTF-8') to convert keys_1_list items for bytes to str ?
    items_list=group['values'][()]
    
    keys_list=[(keys_0,keys_1) for keys_1 in keys_1_list]
    
    #Rebuild LCA_dict
    LCA_dict={}
    
    LCA_dict=dict(zip(keys_list,items_list))
    
    return LCA_dict;


#function to create list to store in hfd5 for LCA object _***_dict: _biosphere_dict, _activity_dict, _product_dict
def _LCA_dict_to_hdf5(LCA_dict,hdf5_file,group_path):
    
    # Retrieve or create the groups and subgroups
    group=hdf5_file.require_group(group_path)  
    
    keys=[key for key in LCA_dict.keys()]
    values=[item for item in LCA_dict.values()]
    
    # Create datasets containing values of csr matrix
    group.create_dataset('keys',data=keys,compression="gzip")
    group.create_dataset('values',data=values,compression="gzip")

    return;

#function to rebuild LCA object _***_dict: _biosphere_dict, _activity_dict, _product_dict
def _hdf5_to_LCA_dict(hdf5_file,group_path):
    
    # Retrieve or create the groups and subgroups
    group=hdf5_file.require_group(group_path)  
    
    keys=group['keys'][()]
    values=group['values'][()]
    
    #Rebuild LCA_dict
    LCA_dict={}
    
    LCA_dict=dict(zip(keys,values))

    return LCA_dict;


#Function to write csr matrix, _dict from LCA objects and any numpy.ndarray

def write_LCA_obj_to_HDF5_file(LCA_obj,hdf5_file,group_path):
    
    dict_names_to_check=['biosphere_dict', 'activity_dict', 'product_dict']
    _dict_names_to_check=['_biosphere_dict', '_activity_dict', '_product_dict']
    
    #If object = A or B matrix
    if type(LCA_obj)==sp.sparse.csr.csr_matrix:
        #csr_matrix_to_hdf5(LCA_obj,hdf5_file,group_path)
        
        #### Direct copy of the function because the call to the function does not work... --> Works now!
        # Retrieve or create groups and subgroups
        group=hdf5_file.require_group(group_path)

        # Create datasets containing values of csr matrix
        group.create_dataset('data',data=LCA_obj.data,compression="gzip",dtype=np.float32)
        group.create_dataset('indptr',data=LCA_obj.indptr,compression="gzip")
        group.create_dataset('indices',data=LCA_obj.indices,compression="gzip")
        group.create_dataset('shape',data=LCA_obj.shape,compression="gzip")
        ######
        
    #If object = _***_dict
    elif group_path.rsplit('/', 1)[1] in _dict_names_to_check:
        _LCA_dict_to_hdf5(LCA_obj,hdf5_file,group_path)    
            
    
    #If object = ***_dict
    elif group_path.rsplit('/', 1)[1] in dict_names_to_check:
        LCA_dict_to_hdf5(LCA_obj,hdf5_file,group_path)
        
    else:
        
        #store as float32 if type is float64 to save space
        if LCA_obj.dtype == np.dtype('float64'):
            hdf5_file.create_dataset(group_path,data=LCA_obj,compression="gzip",dtype=np.float32)
            
        else:
            hdf5_file.create_dataset(group_path,data=LCA_obj,compression="gzip")
            
    
    return;


def h5py_dataset_iterator(g, prefix=''):
    for key in g.keys():
        item = g[key]
        path = '{}/{}'.format(prefix, key)
        if isinstance(item, h5py.Dataset): # test for dataset
            yield (path, item)
        elif isinstance(item, h5py.Group): # test for group (go down)
            yield from h5py_dataset_iterator(item, path)

            
def append_scalar_to_list_hdf5_dataset(hdf5_file,dataset_path,value):
    
    try:
        dataset=hdf5_file[dataset_path]
        dataset.resize((dataset.shape[0]+1,))
        dataset[-1] = value
    except:
        hdf5_file.create_dataset(dataset_path,shape=(1,), maxshape=(None,))
        dataset=hdf5_file[dataset_path]
        dataset[-1] = value
        
    return;


#######################################
# Dependant LCI Monte Carlo functions #
#######################################


#Clean the HDF5 file with LCA MC results for not complete iterations (i.e. not all activities calculated for the last iteration)
def clean_hdf5_file_MC_LCA_results(hdf5_file_MC_results,worker_id):
    
    complete_iterations=int(hdf5_file_MC_results.attrs['Number of complete iterations'])
    
    if "LCA_Dependant_Monte_Carlo_aggregated_results" in hdf5_file_MC_results.filename:
        for (dataset_path, dset) in h5py_dataset_iterator(hdf5_file_MC_results):
            if '/lci_iteration_name_list' not in dataset_path:
                results_iterations=dset.size
                if results_iterations!=complete_iterations:
                    dataset.resize((dataset.shape[0]-1,))
                    print('--Incomplete iterations for LCA results removed for worker {} for {}'.format(worker_id,dataset_path))
                    
                
    elif "LCA_Dependant_Monte_Carlo_disaggregated_results" in hdf5_file_MC_results.filename:
        for uncertainty_level in hdf5_file_MC_results.items:
            if 'lci_iteration_name_list' not in uncertainty_level[0]:
                for act in uncertainty_level[1]:
                    for impact_method in act[1]:
                        results_iterations=len(impact_method[1])
                        if results_iterations!=complete_iterations:
                            iteration_name_to_delete=results_iterations-1
                            del impact_method[1][str(iteration_name_to_delete)]
                            print('--Incomplete iterations for LCA results removed for worker {} for activity {} and impact method'.format(worker_id,act[0],impact_method[0]))
                            
    return;


#Create a file that gather all MC results and Useful info in one file
def gathering_MC_results_in_one_hdf5_file(path_for_saving,results_disaggregated_or_not):
    
    disaggregated_root_file_name='LCA_Dependant_Monte_Carlo_disaggregated_results'
    aggregated_root_file_name='LCA_Dependant_Monte_Carlo_aggregated_results'
    
    root_file_name_list=[aggregated_root_file_name,disaggregated_root_file_name]
    
    for root_file_name in root_file_name_list:
    
    
    #Create the gathering file
    if results_disaggregated_or_not == "disaggregated":
        hdf5_file_all_MC_results=h5py.File(path_for_saving+'\\'+disaggregated_root_file_name+'_ALL.hdf5','w-')
        
    else:
        hdf5_file_all_MC_results=h5py.File(path_for_saving+'\\'+aggregated_root_file_name+'_ALL.hdf5','w-')
    
    #Retrieve child file paths
    child_hdf5_file_paths = [os.path.join(path_for_saving,fn) for fn in next(os.walk(path_for_saving))[2] if '.hdf5' in fn]
    child_MC_results_paths=[path for path in child_hdf5_file_paths if 'LCI_Dependant_Monte_Carlo_results_worker' in path]
    child_DB_info_paths=[path for path in child_hdf5_file_paths if 'Useful_info_per_DB' in path]
    
    #Gathering MC results
    complete_iterations=0
    
    for child_file_path in child_MC_results_paths:
        
        #Clean incomplete iterations before gathering 
        child_hdf5_file=h5py.File(child_file_path,'a')
        clean_hdf5_file_MC_results(child_hdf5_file,child_file_path.rsplit('\\', 1)[1])
        child_hdf5_file.close()
        
        child_hdf5_file=h5py.File(child_file_path,'r')
    
        for (child_dataset_path, dset) in h5py_dataset_iterator(child_hdf5_file):
            
            #For supply array
            if 'supply_array' in child_dataset_path:
            
                #Create similar path for the master file
                child_iteration_name=child_dataset_path.rsplit('/', 1)[1]
                master_iteration_name=str(int(child_iteration_name)+complete_iterations)
                master_dataset_path=child_dataset_path.rsplit('/', 1)[0]+'/'+master_iteration_name

                #Link child data to master file
                hdf5_file_all_MC_results[master_dataset_path] = h5py.ExternalLink(child_file_path, child_dataset_path)
             
            #For A and B matrix
            else:
                #Create similar path for the master file
                child_iteration_name=child_dataset_path.rsplit('/', 2)[1]
                master_iteration_name=str(int(child_iteration_name)+complete_iterations)
                master_dataset_path=child_dataset_path.rsplit('/', 2)[0]+'/'+master_iteration_name+'/'+child_dataset_path.rsplit('/', 2)[2]

                #Link child data to master file
                hdf5_file_all_MC_results[master_dataset_path] = h5py.ExternalLink(child_file_path, child_dataset_path)

        complete_iterations=complete_iterations+int(child_hdf5_file.attrs['Number of complete iterations'])
        child_hdf5_file.close()
        
    hdf5_file_all_MC_results.attrs['Number of complete iterations']=complete_iterations
    
    #Useful info
    for child_file_path in child_DB_info_paths:
        
        child_hdf5_file=h5py.File(child_file_path,'r')
    
        for (child_dataset_path, dset) in h5py_dataset_iterator(child_hdf5_file):
            
            #Create the master path
            master_dataset_path=child_dataset_path
            
            #Link child data to master file
            hdf5_file_all_MC_results[master_dataset_path] = h5py.ExternalLink(child_file_path, child_dataset_path)
        
        db_name=child_hdf5_file.attrs['Database name']
        child_hdf5_file.close()
        
    hdf5_file_all_MC_results.attrs['Database name']=db_name
    
    hdf5_file_all_MC_results.close()
        
    return;
    
    




#Dependant LCA Monte Carlo for each activity and functional unit defined in functional_units = [{act.key: 1}]
def worker_process(project, 
                   job_id, 
                   worker_id, 
                   iterations,
                   functional_units,
                   collector_functional_unit,
                   hdf5_file_MC_LCI_results_path,
                   hdf5_file_deterministic_lci_path,
                   hdf5_file_MC_LCA_results_aggregated_path,
                   hdf5_file_MC_LCA_results_disaggregated_path,
                   impact_method_name_list,
                   results_disaggregated_or_not):
    
    projects.set_current(project)
    
    #Open the HDF5 files for each worker to write LCA results
    hdf5_file_MC_LCI_results=h5py.File(hdf5_file_MC_LCI_results_path,'r')
    lci_complete_iterations=int(hdf5_file_MC_LCI_results.attrs['Number of complete iterations'])
    
    if results_disaggregated_or_not == "disaggregated":
        hdf5_file_MC_LCA_results=h5py.File(hdf5_file_MC_LCA_results_disaggregated_path,'a')
    else:
        hdf5_file_MC_LCA_results=h5py.File(hdf5_file_MC_LCA_results_aggregated_path,'a')
    
    hdf5_file_deterministic_lci=h5py.File(hdf5_file_deterministic_lci_path,'r')
    
    #Retrieve or set the number of LCA complete iterations (all activities calculated for one iteration)
    try:
        lca_complete_iterations=int(hdf5_file_MC_LCA_results.attrs['Number of complete iterations'])
        print('--Previous number of complete iterations for worker {} is {}'.format(worker_id, lca_complete_iterations))
    except:
        hdf5_file_MC_LCA_results.attrs['Number of complete iterations']=0
        lca_complete_iterations=0
        
    
    #Clean incomplete iterations if needed
    if lca_complete_iterations>0:
        clean_hdf5_file_MC_LCA_results(hdf5_file_MC_LCA_results,worker_id)
    
    
    #Retrieve biosphere_dict and activity_dict
    _biosphere_dict=_hdf5_to_LCA_dict(hdf5_file_MC_LCI_results,'/_biosphere_dict')
    
    if results_disaggregated_or_not == "disaggregated":
        activity_dict=hdf5_to_LCA_dict(hdf5_file_MC_LCI_results,'/activity_dict')
    
    
    #Construct the characterization useful info (cf_params and cf_rng) for all LCIA methods
    impact_method_dict={}
    
    for impact_method_name in impact_method_name_list:
        
        method_filepath = [Method(impact_method_name).filepath_processed()]

        cf_params, _, _, characterization_matrix = MatrixBuilder.build(
                    method_filepath,
                    "amount",
                    "flow",
                    "row",
                    row_dict=_biosphere_dict,
                    one_d=True,
        )

        cf_rng = MCRandomNumberGenerator(cf_params, seed=None)
        
        impact_method_dict[impact_method_name]={}
        impact_method_dict[impact_method_name]['cf_params']=cf_params
        impact_method_dict[impact_method_name]['cf_rng']=cf_rng
        impact_method_dict[impact_method_name]['characterization_matrix_deterministic']=characterization_matrix
    
   
       
    #LCA iterations
    for iteration in range(iterations):
        
        #Name of the iteration for the storage, starts from 0
        lca_iteration_name=lca_complete_iterations
        
        #Randomly choose an LCI iteration
        lci_iteration_name=random.randint(0,lci_complete_iterations-1)
        
        #Retrieve B matrix for uncertainty LCI 1
        biosphere_matrix=hdf5_to_csr_matrix(hdf5_file_MC_LCI_results,'/biosphere_matrix/'+str(lci_iteration_name))      
        
        #Regenerate the characterization_matrix for each iteration for all impact methods
        characterization_matrix_dict={}
        
        for impact_method_name in impact_method_dict:
            
            cf_params=impact_method_dict[impact_method_name]['cf_params']
            cf_rng=impact_method_dict[impact_method_name]['cf_rng']
            
            characterization_matrix = MatrixBuilder.build_diagonal_matrix(cf_params, _biosphere_dict,"row", "row", new_data=cf_rng.next())#For disaggregated results
            
            #For disaggregated results
            if results_disaggregated_or_not == "disaggregated":
                characterization_matrix_dict[impact_method_name]=characterization_matrix
            
            #For aggregated results
            else:
                characterization_matrix_array=np.array(characterization_matrix.sum(1))
                characterization_matrix_dict[impact_method_name]=characterization_matrix_array
            
        #Iterations per activity
        for act_index, fu in enumerate(functional_units):

            #Creating UUID for each activity
            actKey = list(fu.keys())[0][1]
            
            
            #Retrieve the inventory for uncertainty LCI 0
            if results_disaggregated_or_not == "disaggregated":
                inventory_lci0_path='/inventory/'+actKey+'/disaggregated'
            else:
                inventory_lci0_path='/inventory/'+actKey+'/aggregated'

            inventory_lci0=hdf5_file_deterministic_lci[inventory_lci0_path][()]
            
            
            #Retrieve supply_array for uncertainty LCI 1
            supply_array=hdf5_file_MC_LCI_results['/supply_array/'+actKey+'/'+str(lci_iteration_name)][()]
            
            #Calculate inventory for uncertainty LCI 1 
            #For disaggregated results --> inventory is a csr matrix
            if results_disaggregated_or_not == "disaggregated":
                count = len(activity_dict)
                inventory = biosphere_matrix * sparse.spdiags([supply_array], [0], count, count) #For disaggregated results
            
            #For aggregated results --> inventory is a vector
            else:
                #inventory_array=np.array(inventory.sum(1)) #For aggregated results

                inventory = biosphere_matrix * supply_array 
            
            
            #Calculate impact scores for all impact categories and Store impact_score
            for impact_method_name in characterization_matrix_dict:
                
                characterization_matrix=characterization_matrix_dict[impact_method_name]
                characterization_matrix_lcia0=impact_method_dict[impact_method_name]['characterization_matrix_deterministic']
                
                impact_score_path='/Uncertainty LCI 1 LCIA 1/'+actKey+'/'+str(impact_method_name)
                impact_score_lcia0_path='/Uncertainty LCI 1 LCIA 0/'+actKey+'/'+str(impact_method_name)
                impact_score_lci0_lcia1_path='/Uncertainty LCI 0 LCIA 1/'+actKey+'/'+str(impact_method_name)
                
                #For disaggregated results --> impact_score is a csr matrix of impact scores 
                #disaggregated per direct contributing activity and direct contributing elementary flows
                if results_disaggregated_or_not == "disaggregated":
                    
                    #Uncertainty LCI 1 LCIA 1
                    impact_score= characterization_matrix * inventory
                    impact_score_path=impact_score_path+'/'+str(lca_iteration_name)
                    write_LCA_obj_to_HDF5_file(impact_score,hdf5_file_MC_LCA_results,impact_score_path)
                    
                    #Uncertainty LCI 1 LCIA 0
                    impact_score_lcia0= characterization_matrix_lcia0 * inventory
                    impact_score_lcia0_path=impact_score_lcia0_path+'/'+str(lca_iteration_name)
                    write_LCA_obj_to_HDF5_file(impact_score_lcia0,hdf5_file_MC_LCA_results,impact_score_lcia0_path)
                    
                    #Uncertainty LCI 0 LCIA 1
                    impact_score_lci0_lcia1= characterization_matrix * inventory_lci0
                    impact_score_lci0_lcia1_path=impact_score_lci0_lcia1_path+'/'+str(lca_iteration_name)
                    write_LCA_obj_to_HDF5_file(impact_score_lci0_lcia1,hdf5_file_MC_LCA_results,impact_score_lci0_lcia1_path)
                    
                    #store info on activities and EF?
                    
                #For aggregated results --> impact_score is a scalar, store as vector for each impact method    
                else:
                    #Uncertainty LCI 1 LCIA 1
                    characterization_matrix=np.reshape(characterization_matrix,characterization_matrix.shape[0])
                    impact_score= np.dot(characterization_matrix, inventory)
                    append_scalar_to_list_hdf5_dataset(hdf5_file_MC_LCA_results,impact_score_path,impact_score)
                    
                    #Uncertainty LCI 1 LCIA 0
                    characterization_matrix_lcia0=np.array(characterization_matrix_lcia0.sum(1))
                    characterization_matrix_lcia0=np.reshape(characterization_matrix_lcia0,characterization_matrix_lcia0.shape[0])
                    impact_score_lcia0= np.dot(characterization_matrix_lcia0, inventory)
                    append_scalar_to_list_hdf5_dataset(hdf5_file_MC_LCA_results,impact_score_lcia0_path,impact_score_lcia0)
                    
                    #Uncertainty LCI 0 LCIA 1
                    impact_score_lci0_lcia1= np.dot(characterization_matrix, inventory_lci0)
                    append_scalar_to_list_hdf5_dataset(hdf5_file_MC_LCA_results,impact_score_lci0_lcia1_path,impact_score_lci0_lcia1)
                        
            
        #Store the list of iteration names from LCI
        lci_iteration_name_list_path='/lci_iteration_name_list'
        append_scalar_to_list_hdf5_dataset(hdf5_file_MC_LCA_results,lci_iteration_name_list_path,lci_iteration_name)
        
        
        #Count the number of complete iterations for LCA results
        lca_complete_iterations=lca_iteration_name+1
        hdf5_file_MC_LCA_results.attrs['Number of complete iterations']= lca_complete_iterations
            
                        
    hdf5_file_deterministic_lci.close()
    hdf5_file_MC_LCI_results.close()
    hdf5_file_MC_LCA_results.close()
    
    return;
                        
                
                
        
def get_deterministic_inventory(collector_functional_unit, functional_units, hdf5_file_deterministic_lci, job_id):
        
    #Get metadata for the HDF5 file
    DB_name= list(functional_units[0].keys())[0][0]
    
    #Write useful information to HDF5 file
    hdf5_file_deterministic_lci.attrs['Database name']=DB_name
    hdf5_file_deterministic_lci.attrs['Creation ID']=job_id
    hdf5_file_deterministic_lci.attrs['Description']='HDF5 file containing the deterministic values for inventory for all activities in the database'
    
    #Retrieve B matrix for uncertainty LCI 0 --> use sacrificial LCA
    sacrificial_lca = LCA(collector_functional_unit)
    sacrificial_lca.lci()
    biosphere_matrix_lci0=sacrificial_lca.biosphere_matrix
    count = len(sacrificial_lca.activity_dict)
    
    
    #Retrieve the supply_array for uncertainty LCI 0  
    for act_index, fu in enumerate(functional_units):
        
        #Creating UUID for each activity
        actKey = list(fu.keys())[0][1]
        
        #Calculate the deterministic supply_array
        lca = LCA(fu)
        lca.lci()
        supply_array_lci0=lca.supply_array
        
        #Calculate deterministic inventories. inventory_lci0_disaggregated is a csr matrix.
        inventory_lci0_disaggregated = biosphere_matrix_lci0 * sparse.spdiags([supply_array_lci0], [0], count, count)
        inventory_lci0_aggregated = biosphere_matrix_lci0 * supply_array_lci0
        
        #Store deterministic inventories
        inventory_lci0_disaggregated_path='/inventory/'+actKey+'/disaggregated'
        inventory_lci0_aggregated_path='/inventory/'+actKey+'/aggregated'
        
        write_LCA_obj_to_HDF5_file(inventory_lci0_disaggregated,hdf5_file_deterministic_lci,inventory_lci0_disaggregated_path)
        write_LCA_obj_to_HDF5_file(inventory_lci0_aggregated,hdf5_file_deterministic_lci,inventory_lci0_aggregated_path)
    
    return;    




def Dependant_LCA_Monte_Carlo_results(project, 
                                                 database, 
                                                 iterations, 
                                                 cpus, 
                                                 hdf5_file_MC_LCI_results_path, 
                                                 path_for_saving,
                                                 impact_method_name_list,
                                                 results_disaggregated_or_not):
    
    projects.set_current(project)
    bw2setup()

    #Path the write the results
    BASE_OUTPUT_DIR = path_for_saving

    #ID to identify who and when was the calculation made
    now = datetime.datetime.now()
    job_id = "{}_{}-{}-{}_{}h{}".format(os.environ['COMPUTERNAME'],now.year, now.month, now.day, now.hour, now.minute)

    #Selection of activities for MC analysis
    db = Database(database)
    #activities = [activity for activity in db]
    act1=db.get('e929619f245df590fee5d72dc979cdd4')
    act2=db.get('bdf7116059abfcc6b8b9ade1a641e578')
    act3=db.get('c8c815c68836adaf964daaa001a638a3')
    activities = [act1,act2,act3]
    
    #Create objects to pass the functional units = 1 for each activity
    functional_units = [ {act.key: 1} for act in activities ]
    collector_functional_unit = {k:v for d in functional_units for k, v in d.items()}
    
    #Create and save all the useful information related to the deterministic version of the database
    path_for_saving=BASE_OUTPUT_DIR
    hdf5_file_name="Deterministic_LCI.hdf5"
    hdf5_file_deterministic_lci_path=path_for_saving+"\\"+hdf5_file_name
    
    if os.path.isfile(hdf5_file_deterministic_lci_path)==False:
        hdf5_file_deterministic_lci=h5py.File(hdf5_file_deterministic_lci_path,'a')
        get_deterministic_inventory(collector_functional_unit, functional_units, hdf5_file_deterministic_lci, job_id)
        hdf5_file_deterministic_lci.close()
    
    #Code to slipt the work between each CPUs of the computer (called workers). The work refers here to the dependant LCI MC for each activity 
    workers = []

    for worker_id in range(cpus):
        #Create or open the HDF5 file for each worker and write metadata
        hdf5_file_name="LCA_Dependant_Monte_Carlo_aggregated_results_worker"+str(worker_id)+".hdf5"
        hdf5_file_MC_LCA_results_aggregated_path=BASE_OUTPUT_DIR+"\\"+hdf5_file_name
        hdf5_file_name="LCA_Dependant_Monte_Carlo_disaggregated_results_worker"+str(worker_id)+".hdf5"
        hdf5_file_MC_LCA_results_disaggregated_path=BASE_OUTPUT_DIR+"\\"+hdf5_file_name
        
        if results_disaggregated_or_not == "disaggregated":
            hdf5_file_MC_results=h5py.File(hdf5_file_MC_LCA_results_disaggregated_path,'a')
        else:
            hdf5_file_MC_results=h5py.File(hdf5_file_MC_LCA_results_aggregated_path,'a')
        
        hdf5_file_MC_results.attrs['Database name']=db.name
        hdf5_file_MC_results.attrs['Worker ID']=worker_id
        hdf5_file_MC_results.attrs['Description']='HDF5 file containing all LCA dependant Monte Carlo results per activity/iteration'

        hdf5_file_MC_results.close()

        # Create child processes that can work apart from parent process
        child = mp.Process(target=worker_process, args=(projects.current, 
                                                        job_id, worker_id, 
                                                        iterations,
                                                        functional_units,
                                                        collector_functional_unit,
                                                        hdf5_file_MC_LCI_results_path,
                                                        hdf5_file_deterministic_lci_path,
                                                        hdf5_file_MC_LCA_results_aggregated_path,
                                                        hdf5_file_MC_LCA_results_disaggregated_path,
                                                        impact_method_name_list,
                                                        results_disaggregated_or_not))
        workers.append(child)
        child.start()
        
    return;
      
        
        
#Useful when the code is run from the console to execute the main function
if __name__ == '__main__':
    Dependant_LCA_Monte_Carlo_results(project="iw_integration", 
                                      database="ecoinvent 3.3 cutoff", 
                                      iterations=10, 
                                      cpus=4, 
                                      hdf5_file_MC_LCI_results_path="D:\\Dossiers professionnels\\Logiciels\\Brightway 2\\Test Dependant LCI Monte Carlo - test 3\\LCI_Dependant_Monte_Carlo_results_ALL.hdf5", 
                                      path_for_saving="D:\\Dossiers professionnels\\Logiciels\\Brightway 2\\Test Dependant LCA Monte Carlo",
                                      impact_method_name_list=[("CML 2001", "climate change", "GWP 100a"),("CML 2001", "eutrophication potential", "generic")],
                                      results_disaggregated_or_not="aggregated")
