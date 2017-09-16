from brightway2 import *
import numpy as np
import os
import multiprocessing as mp
from scipy.sparse.linalg import factorized, spsolve
from scipy import sparse
import datetime
import pickle
import click
import scipy as sp
import h5py

"""
Used to generate uncertainty information at the database level.
For each iteration:
- New values for uncertain parameters of the technosphere (A) and biosphere (B) matrices are generated
- Cradle-to-gate LCI results are calculated for all potential output of the LCI database

The following is stored in a specified directory: 
- All values of the A and B matrices
- For each functional unit: 
    - the supply array (aka the scaling vector)
    - the life cycle inventory
""" 

#Change for DirectSolvingMonteCarloLCA(MonteCarloLCA, DirectSolvingMixin)?
class DirectSolvingMonteCarloLCA(MonteCarloLCA, DirectSolvingMixin):
    pass

#Dependant LCI Monte Carlo for each activity and functional unit defined in functional_units = [{act.key: FU}]
def worker_process(project, job_id, worker_id, functional_units, iterations):
    
    #Open the HDF5 file for each worker
    hdf5_file_name="LCI_Dependant_Monte_Carlo_results_worker"+str(worker_id)+".hdf5"
    hdf5_file_MC_results_path=path_for_saving+"\\"+hdf5_file_name
        
    hdf5_file_MC_results=h5py.File(hdf5_file_MC_results_path,'a')
    
    
    ###### Add attributes for metadata!! --> in the main function to write it just once
    

    projects.set_current(project)
    
    #Creating the LCA object --> set fix_dictionaries=False as not useful here?
    lca = DirectSolvingMonteCarloLCA(demand = functional_units[0])
    lca.load_data()
    
    #Create and save objects per iteration
    for iteration in range(iterations):
    
        print('--Starting job for worker {}, iteration {}'.format(worker_id, iteration))        
        
        #Creating A and B matrix
        lca.rebuild_technosphere_matrix(lca.tech_rng.next())
        lca.rebuild_biosphere_matrix(lca.bio_rng.next())
        
        #Saving A and B to HDF5 file
        group_path_techno='/technosphere_matrix/'+str(iteration)
        group_path_bio='/biosphere_matrix/'+str(iteration)
        write_LCA_obj_to_HDF5_file(lca.technosphere_matrix,hdf5_file_MC_results,group_path_techno)
        write_LCA_obj_to_HDF5_file(lca.biosphere_matrix,hdf5_file_MC_results,group_path_bio)
        hdf5_file_MC_results[group_path_techno].attrs['Creation ID']=job_id
        hdf5_file_MC_results[group_path_bio].attrs['Creation ID']=job_id

        #For calculation
        lca.decompose_technosphere()
        
        #Create and save objects per activity/iteration
        for act_index, fu in enumerate(functional_units):
            
            #Creating UUID for each activity
            actKey = list(fu.keys())[0][1]
            
            #Create demand_array
            lca.build_demand_array(fu)

            #Create supply_array
            lca.supply_array = lca.solve_linear_system()
            
            #Save supply_array to HDF5 file
            group_path_supply='/supply_array/'+actKey+'/'+str(iteration)
            write_LCA_obj_to_HDF5_file(lca.supply_array,hdf5_file_MC_results,group_path_supply)
            hdf5_file_MC_results[group_path_supply].attrs['Creation ID']=job_id
            hdf5_file_MC_results['/supply_array/'+actKey].attrs['Number of iterations']=iteration
            
            
    hdf5_file_MC_results.close()
        
    return;
    
def get_useful_info(collector_functional_unit, hdf5_file_useful_info_per_DB, job_id, activities):

    # Sacrificial LCA to extract relevant information (demand of 1 for all activities)
    # Done on the "collector" functional unit to ensure that all activities and 
    # exchanges are covered in the common dicts (only relevant if some activities 
    # link to other upstream databases
    sacrificial_lca = LCA(collector_functional_unit)
    sacrificial_lca.lci()
    
    #Get data to store 
    product_dict=sacrificial_lca.product_dict
    biosphere_dict=sacrificial_lca.biosphere_dict
    activity_dict=sacrificial_lca.activity_dict
    tech_params=sacrificial_lca.tech_params
    bio_params=sacrificial_lca.bio_params
    activities_keys=[act.key[1] for act in activities]
    rev_activity_dict, rev_product_dict, rev_bio_dict = sacrificial_lca.reverse_dict()
    
    #Get metadata for the HDF5 file
    DB_name=activities[0].key[0]
    
    #Write useful information to HDF5 file
    hdf5_file_useful_info_per_DB.attrs['Database name']=DB_name
    hdf5_file_useful_info_per_DB.attrs['Creation ID']=job_id
    hdf5_file_useful_info_per_DB.attrs['Description']='HDF5 file containing all useful information related to the database in order to use dependant Monte Carlo results'
    
    write_LCA_obj_to_HDF5_file(product_dict,hdf5_file_useful_info_per_DB,'/product_dict')
    write_LCA_obj_to_HDF5_file(biosphere_dict,hdf5_file_useful_info_per_DB,'/biosphere_dict')
    write_LCA_obj_to_HDF5_file(activity_dict,hdf5_file_useful_info_per_DB,'/activity_dict')
    write_LCA_obj_to_HDF5_file(tech_params,hdf5_file_useful_info_per_DB,'/tech_params')
    write_LCA_obj_to_HDF5_file(bio_params,hdf5_file_useful_info_per_DB,'/bio_params')
    write_LCA_obj_to_HDF5_file(activities_keys,hdf5_file_useful_info_per_DB,'/activities_keys')
    write_LCA_obj_to_HDF5_file(rev_activity_dict,hdf5_file_useful_info_per_DB,'/rev_activity_dict')
    write_LCA_obj_to_HDF5_file(rev_product_dict,hdf5_file_useful_info_per_DB,'/rev_product_dict')
    write_LCA_obj_to_HDF5_file(rev_bio_dict,hdf5_file_useful_info_per_DB,'/rev_bio_dict')

    return None;


#Create and save useful information during Dependant LCI MC : database objects (_dict, activities, _params, reverse_dict), iteration objects (_sample, i.e. A and B _matrix), act/iteration objects (supply_array)    
def Dependant_LCI_Monte_Carlo(project, database, iterations, cpus, output_dir):

    projects.set_current(project)
    bw2setup()
    
    #Path the write the results
    BASE_OUTPUT_DIR = output_dir
    
    #ID to identify who and when was the calculation made
    now = datetime.datetime.now()
    job_id = "{}_{}-{}-{}_{}h{}".format(os.environ['COMPUTERNAME'],now.year, now.month, now.day, now.hour, now.minute)
    
    #Selection of activities for MC analysis
    db = Database(database)
    activities = [activity for activity in db]
    
    #Create objects to pass the functional units = 1 for each activity
    functional_units = [ {act.key: 1} for act in activities ]
    collector_functional_unit = {k:v for d in functional_units for k, v in d.items()}
    
    #Create or open the HDF5 file for useful information storage per DB
    path_for_saving=BASE_OUTPUT_DIR
    hdf5_file_name="Useful_info_per_DB.hdf5"
    hdf5_file_useful_info_per_DB_path=path_for_saving+"\\"+hdf5_file_name
    
    hdf5_file_useful_info_per_DB=h5py.File(hdf5_file_useful_info_per_DB_path,'a')
    
    #Create and save all the useful information related to the database only
    ##os.chdir(BASE_OUTPUT_DIR)
    ##job_dir = os.path.join(BASE_OUTPUT_DIR, job_id)
    ##os.mkdir(job_dir)
    
    get_useful_info(collector_functional_unit, hdf5_file_useful_info_per_DB, job_id, activities)
    
    #Code to slipt the work between each CPUs of the computer (called workers). The work refers here to the dependant LCI MC for each activity 
    workers = []

    for worker_id in range(cpus):
        #Create or open the HDF5 file for each worker and write metadata
        hdf5_file_name="LCI_Dependant_Monte_Carlo_results_worker"+str(worker_id)+".hdf5"
        hdf5_file_MC_results_path=path_for_saving+"\\"+hdf5_file_name
            
        hdf5_file_MC_results=h5py.File(hdf5_file_MC_results_path,'a')
            
        hdf5_file_MC_results.attrs['Database name']=db.name
        hdf5_file_MC_results.attrs['Worker ID']=worker_id
        hdf5_file_MC_results.attrs['Description']='HDF5 file containing all dependant Monte Carlo results per iteration (A and B matrix) and per activity/iteration (supply_array)'
            
        hdf5_file_MC_results.close()
            
        # Create child processes that can work apart from parent process
        child = mp.Process(target=worker_process, args=(projects.current, job_id, worker_id, functional_units, iterations))
        workers.append(child)
        child.start()

    return;
      
        
        
#Useful when the code is run from the console to execute the main function
if __name__ == '__main__':
    Dependant_LCI_Monte_Carlo()

#Useful when the code is run from the console to pass arguments to the main function
@click.command()
@click.option('--project', default='default', help='Brightway2 project name', type=str)
@click.option('--database', help='Database name', type=str)
@click.option('--iterations', default=1000, help='Number of Monte Carlo iterations', type=int)
@click.option('--cpus', default=mp.cpu_count(), help='Number of used CPU cores', type=int)
@click.option('--output_dir', help='Output directory path', type=str)






##################
# HDF5 functions #
##################

# function create a group containing all the information of a csr matrix scipy  

def csr_matrix_to_hdf5(csr,hdf5_file,group_path):
    
    # Retrieve or create groups and subgroups
    group=hdf5_file.require_group(group_path)
    
    # Create datasets containing values of csr matrix
    group.create_dataset('data',data=csr.data,compression="gzip",dtype=np.float32)
    group.create_dataset('indptr',data=csr.indptr,compression="gzip")
    group.create_dataset('indices',data=csr.indices,compression="gzip")
    group.attrs['shape']=csr.shape
    
    return;


#function to create list to store in hfd5 for LCA object _dict: biosphere_dict, activity_dict, product_dict

def LCA_dict_to_hdf5(LCA_dict,hdf5_file,group_path):
    
    # Retrieve or create the groups and subgroups
    group=hdf5_file.require_group(group_path)
    
    keys_list=[int(key) for key in LCA_dict.keys()]
    items_list=[item for item in LCA_dict.values()]
    
    # Create datasets containing values of csr matrix
    group.create_dataset('keys',data=keys_list,compression="gzip")
    group.create_dataset('values',data=items_list,compression="gzip")

    
    return;


#Function to write csr matrix, _dict from LCA objects and any numpy.ndarray

def write_LCA_obj_to_HDF5_file(LCA_obj,hdf5_file,group_path):
    
    dict_names_to_check=['biosphere_dict', 'activity_dict', 'product_dict']
    
    #If object = A or B matrix
    if type(LCA_obj)==sp.sparse.csr.csr_matrix:
        csr_matrix_to_hdf5(LCA_obj,hdf5_file,group_path)
    
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


