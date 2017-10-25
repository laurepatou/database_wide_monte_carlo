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



##################
# HDF5 functions #
##################

# All those functions work based on LCA objects from Brightway

# function create a group containing all the information of a csr matrix scipy  

def csr_matrix_to_hdf5(csr,hdf5_file,group_path):
    
    # Retrieve or create groups and subgroups
    group=hdf5_file.require_group(group_path)
    
    csr_size=csr.nnz
    
    # Create datasets containing values of csr matrix
    group.create_dataset('data',data=csr.data,compression="gzip",dtype=np.float32)
    group.create_dataset('indptr',data=csr.indptr,compression="gzip")
    group.create_dataset('indices',data=csr.indices,compression="gzip")
    group.create_dataset('shape',data=csr.shape)

    
    return;


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
        group.create_dataset('shape',data=LCA_obj.shape)

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


#######################################
# Dependant LCI Monte Carlo functions #
#######################################


#Change for DirectSolvingMonteCarloLCA(MonteCarloLCA, DirectSolvingMixin)?
class DirectSolvingMonteCarloLCA(MonteCarloLCA, DirectSolvingMixin):
    pass

#Clean the HDF5 file with LCI MC results for not complete iterations (i.e. not all activities calculated for the last iteration)
def clean_hdf5_file_MC_results(hdf5_file_MC_results,worker_id):
    
    complete_iterations=int(hdf5_file_MC_results.attrs['Number of complete iterations'])
    
    #Clean techno and bio matrix for not complete iterations
    techno_group=hdf5_file_MC_results['/technosphere_matrix']
    bio_group=hdf5_file_MC_results['/biosphere_matrix']
    techno_iterations=len(techno_group)
    bio_iterations=len(bio_group)

    print('--Number of iterations for A and B retrieved for worker {}'.format(worker_id))

    if techno_iterations!=complete_iterations:
        iteration_name_to_delete=techno_iterations-1
        del techno_group[str(iteration_name_to_delete)]
        #print('--Incomplete iterations for A removed for worker {}'.format(worker_id))

    if bio_iterations!=complete_iterations:
        iteration_name_to_delete=bio_iterations-1
        del bio_group[str(iteration_name_to_delete)]
        #print('--Incomplete iterations for B removed for worker {}'.format(worker_id))

    #Clean supply arrays for not complete iterations
    supply_array_group=hdf5_file_MC_results['/supply_array']

    print('--Number of iterations for supply_array retrieved for worker {}'.format(worker_id))

    for act in supply_array_group:
        supply_act_iterations=len(supply_array_group[act])
        if supply_act_iterations!=complete_iterations:
            iteration_name_to_delete=supply_act_iterations-1
            del supply_array_group[act][str(iteration_name_to_delete)]
            #print('--Incomplete iterations for supply_array removed for worker {} for activity {}'.format(worker_id,act))
            
    return;

#Create a file that gather all MC results and Useful info in one file
def gathering_MC_results_in_one_hdf5_file(path_for_saving):
    
    
    #Create the gathering file
    hdf5_file_all_MC_results_path=os.path.join(path_for_saving,'LCI_Dependant_Monte_Carlo_results_ALL.hdf5')
    hdf5_file_all_MC_results=h5py.File(hdf5_file_all_MC_results_path,'w-')
    
    #Retrieve child file paths
    child_hdf5_file_paths = [os.path.join(path_for_saving,fn) for fn in next(os.walk(path_for_saving))[2] if '.hdf5' in fn]
    child_MC_results_paths=[path for path in child_hdf5_file_paths if 'LCI_Dependant_Monte_Carlo_results_worker' in path]
    child_DB_info_paths=[path for path in child_hdf5_file_paths if 'Useful_info_per_DB' in path]
    
    #Gathering MC results
    complete_iterations=0
    
    for child_file_path in child_MC_results_paths:
        
        worker_id=child_file_path.rsplit('LCI_Dependant_Monte_Carlo_results_worker', 1)[1]
        
        start_1 = time.time()
        
        #Clean incomplete iterations before gathering 
        child_hdf5_file=h5py.File(child_file_path,'r+')
        clean_hdf5_file_MC_results(child_hdf5_file,worker_id)
        child_hdf5_file.close()
        
        end_1 = time.time()
        print("Cleaning in {} secondes for entire DB for worker {}".format(end_1 - start_1,worker_id)) 

        start_2 = time.time()
        
        child_hdf5_file=h5py.File(child_file_path,'r')
    
        for (child_dataset_path, dset) in h5py_dataset_iterator(child_hdf5_file):
            
            #For supply array
            if 'supply_array' in child_dataset_path:
            
                #Create similar path for the master file
                child_iteration_name=child_dataset_path.rsplit('/', 1)[1]
                master_iteration_name=str(int(child_iteration_name)+complete_iterations)
                master_dataset_path='{}/{}'.format(child_dataset_path.rsplit('/', 1)[0],master_iteration_name)

                #Link child data to master file
                hdf5_file_all_MC_results[master_dataset_path] = h5py.ExternalLink(child_file_path, child_dataset_path)
             
            #For A and B matrix
            else:
                #Create similar path for the master file
                child_iteration_name=child_dataset_path.rsplit('/', 2)[1]
                master_iteration_name=str(int(child_iteration_name)+complete_iterations)
                master_dataset_path='{}/{}/{}'.format(child_dataset_path.rsplit('/', 2)[0],master_iteration_name,child_dataset_path.rsplit('/', 2)[2])

                #Link child data to master file
                hdf5_file_all_MC_results[master_dataset_path] = h5py.ExternalLink(child_file_path, child_dataset_path)

        complete_iterations=complete_iterations+int(child_hdf5_file.attrs['Number of complete iterations'])
        child_hdf5_file.close()
        
        end_2 = time.time()
        print("Gathering information in {} secondes for entire DB for {} iterations".format(end_2 - start_2,complete_iterations)) 

        
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
   

#######Functions to calculate correlated LCI######

def calculate_ratio_corr_lognormal(gmean_Xi,gmean_Yj,gsigma_Xi,gsigma_Yj):
    
    #We assume that Xi and Yj are perfeclty correlated
    gsigma_XiYj=gsigma_Xi*gsigma_Yj
    
    gmean_Aij=gmean_Xi-gmean_Yj
    gsigma_Aij=np.sqrt((gsigma_Xi**2)+(gsigma_Yj**2)-(2*gsigma_XiYj))
    
    if np.isnan(gsigma_Aij) or gsigma_Aij==0:
        gsigma_Aij=0.0000000000001
    
    return {'loc':gmean_Aij,'scale':gsigma_Aij,'uncertainty_type':2};


#Only once for the DB
def calculate_ratio_corr_lognormal_for_DB(corr_ef_code_input,
                                          corr_ef_code_output,
                                          corr_twin_ef_input_output,
                                          biosphere_dict,
                                          bio_params,biosphere_names):
    
    #corr_bio_params=bio_params
    ratio_rv_dict_per_activity={}
    inv_biosphere_dict={indice: ef for ef, indice in biosphere_dict.items()}
    
    set_efs=set([key[1] for key in biosphere_dict.keys()])
    
    #Isolate bio_params of interest: from corr_ef_code and with lognormal uncertainty
    corr_input_indices=[biosphere_dict[('biosphere3',code)] for code in corr_ef_code_input if code in set_efs]
    corr_output_indices=[biosphere_dict[('biosphere3',code)] for code in corr_ef_code_output if code in set_efs]
    corr_twin_indices=[(biosphere_dict[('biosphere3',code[0])],biosphere_dict[('biosphere3',code[1])]) for code in corr_twin_ef_input_output if (code[0] in set_efs and code[1] in set_efs)]
    corr_input_twin_indices=[indice[0] for indice in corr_twin_indices]
    corr_output_twin_indices=[indice[1] for indice in corr_twin_indices]
    corr_twin_indices_dict={indice[0]:indice[1] for indice in corr_twin_indices}
    
    #print(corr_twin_indices_dict)
    
    set_input=set(corr_input_indices)
    set_output=set(corr_output_indices)
    set_input_output=set(corr_input_indices+corr_output_indices)
    
    set_twin=set(corr_twin_indices)
    set_input_twin=set(corr_input_twin_indices)
    set_output_twin=set(corr_output_twin_indices)
    set_input_output_twin=set(corr_input_twin_indices+corr_output_twin_indices)
    
    
    selected_bio_params=np.array([bio_param for bio_param in bio_params if (bio_param['row'] in set_input_output and bio_param['uncertainty_type']==2)])
    
    #Per activity
    set_col_indices=list(set(selected_bio_params['col']))
    
    for col in set_col_indices:
        
        act_bio_params=np.array([bio_param for bio_param in selected_bio_params if bio_param['col']==col])
        
        #Test if {input}=0 or {output}=0 --> exclude the activity from the list
        act_input_params=np.array([bio_param for bio_param in act_bio_params if bio_param['row'] in set_input])
        act_output_params=np.array([bio_param for bio_param in act_bio_params if bio_param['row'] in set_output])
        
        try:
            act_input_params[0]
        except IndexError:
            continue
            
        try:
            act_output_params[0]
        except IndexError:
            continue
            
        
        #Test if input=output for twin ef --> remove uncertainty from the bio_params
        #act_input_twin_params=np.array([bio_param for bio_param in act_input_params if (bio_param['row'] in set_input_twin and bio_param['row'] in set_input_output_twin)])
        #act_output_twin_params=np.array([bio_param for bio_param in act_output_params if (bio_param['row'] in set_output_twin and bio_param['row'] in set_input_output_twin)])
        #
        #try:
        #    act_input_twin_params[0]
        #    
        #    for twin_input_row in act_input_twin_params['row']:
        #        twin_output_row=corr_twin_indices_dict[twin_input_row]
        #        
        #        output_twin_params=np.array([bio_param for bio_param in act_output_twin_params if bio_param['row']==twin_output_row])
        #        
        #        try:
        #            output_twin_param=output_twin_params[0]
        #            input_twin_param=np.array([bio_param for bio_param in act_input_twin_params if bio_param['row']==twin_input_row])[0]
        #            
        #            input_row_col_to_remove=(input_twin_param['row'],input_twin_param['col'])
        #            output_row_col_to_remove=(output_twin_param['row'],output_twin_param['col'])
        #                                
        #            if input_twin_param['scale']==output_twin_param['scale']:                        
        #                input_condition=np.where((corr_bio_params['row'] == input_row_col_to_remove[0]) * (corr_bio_params['col'] == input_row_col_to_remove[1]))
        #                corr_bio_params[input_condition[0].tolist()[0]]['uncertainty_type']=1
        #                act_input_params=[bio_param for bio_param in act_input_params if (bio_param['row'] != input_row_col_to_remove[0] and bio_param['col'] != input_row_col_to_remove[1])]
        #                
        #                output_condition=np.where((corr_bio_params['row'] == output_row_col_to_remove[0]) * (corr_bio_params['col'] == output_row_col_to_remove[1]))
        #                corr_bio_params[output_condition[0].tolist()[0]]['uncertainty_type']=1
        #                act_output_params=[bio_param for bio_param in act_output_params if (bio_param['row'] != output_row_col_to_remove[0] and bio_param['col'] != output_row_col_to_remove[1])]
        #                
        #                print(col,biosphere_names[inv_biosphere_dict[input_row_col_to_remove[0]]],biosphere_names[inv_biosphere_dict[output_row_col_to_remove[0]]])
        #                
        #        except IndexError:
        #            pass
        #    
        #except IndexError:
        #    pass
            
        
        #Calculate Aij ratio lognormal rv and Store Aij and associate it with Yj only
        for input_param in act_input_params:
            for output_param in act_output_params:
                gmean_Xi=input_param['loc']
                gmean_Yj=output_param['loc']
                gsigma_Xi=input_param['scale']
                gsigma_Yj=output_param['scale']
                params_Aij_dict=calculate_ratio_corr_lognormal(gmean_Xi,gmean_Yj,gsigma_Xi,gsigma_Yj)
                ratio_rv_dict_per_activity[col]={output_param['row']:{input_param['row']:params_Aij_dict}}
                
                #print(type(params_Aij_dict['scale']))
                
                if np.isnan(params_Aij_dict['scale']):
                    print(col,biosphere_names[inv_biosphere_dict[input_param['row']]],biosphere_names[inv_biosphere_dict[output_param['row']]])
                                                 
    return ratio_rv_dict_per_activity; #return (ratio_rv_dict_per_activity,corr_bio_params);

#Each iteration after regenerating the B matrix
def generate_corr_biosphere_matrix(ratio_rv_dict_per_activity,biosphere_dict, biosphere_matrix):
    
    corr_biosphere_matrix=biosphere_matrix.copy()
    
    #per activity
    for col,A in ratio_rv_dict_per_activity.items():
        
        #per output Yj
        for output_row,Aj in A.items():
            
            #print(Aj)

            #Retrieve sum of estimation of Xi
            input_rows=Aj.keys()
            sum_xi=sum([biosphere_matrix[row,col] for row in input_rows])

            #Generate Aij
            Aij_rvs = stats_arrays.UncertaintyBase.from_dicts(*Aj.values())
            Aij_rng = stats_arrays.MCRandomNumberGenerator(Aij_rvs)
            aij=next(Aij_rng)
            sum_aij=sum(aij)
                
            #Calculate Yj
            yj=sum_xi/sum_aij

            #Replace Yj values in the biosphere_matrix
            corr_biosphere_matrix[output_row,col]=yj
    
    return corr_biosphere_matrix;

#Dependant LCI Monte Carlo for each activity and functional unit defined in functional_units = [{act.key: FU}]
def worker_process(project, job_id, worker_id, functional_units, iterations,path_for_saving,collector_functional_unit):
    
    #Open the HDF5 file for each worker
    hdf5_file_name="LCI_Dependant_Monte_Carlo_results_worker{}.hdf5".format(str(worker_id))
    hdf5_file_MC_results_path="{}\\{}".format(path_for_saving,hdf5_file_name)
        
    hdf5_file_MC_results=h5py.File(hdf5_file_MC_results_path,'a')
    
    #for LCI physical correlation: water and land transformation
    projects.set_current(project)
    biosphere3=Database('biosphere3')
    
    lca_1 = LCA(collector_functional_unit)
    lca_1.lci()
    
    land_transfo_efs=[ef for ef in biosphere3 if 'Transformation, ' in str(ef)]
    land_transfo_input={ef['name'].rsplit('Transformation, from ',1)[1]:ef for ef in land_transfo_efs if ', from' in str(ef)}
    land_transfo_output={ef['name'].rsplit('Transformation, to ',1)[1]:ef for ef in land_transfo_efs if ', to' in str(ef)}
    corr_twin_land_transfo=[(land_transfo_input[end_name]['code'],land_transfo_output[end_name]['code']) for end_name in land_transfo_input.keys()]
    land_transfo_code_input=[ef['code'] for ef in land_transfo_input.values()]
    land_transfo_code_output=[ef['code'] for ef in land_transfo_output.values()]
    
    water_efs=[ef for ef in biosphere3 if 'Water' in str(ef['name'])]
    water_code_input=[ef['code'] for ef in water_efs if 'resource' in str(ef['categories'])]
    water_code_output=[ef['code'] for ef in water_efs if 'resource' not in str(ef['categories'])]
    corr_twin_water=[]
    
    biosphere_dict=lca_1.biosphere_dict
    bio_params=lca_1.bio_params
    biosphere_names={('biosphere3',ef['code']):ef['name'] for ef in biosphere3}    
    
    try:

        #Retrieve the number of complete iterations (all activities calculated for one iteration)
        try:
            complete_iterations=int(hdf5_file_MC_results.attrs['Number of complete iterations'])
            print('--Previous number of complete iterations for worker {} is {}'.format(worker_id, complete_iterations))
        except:
            hdf5_file_MC_results.attrs['Number of complete iterations']=0
            complete_iterations=0

        #Clean the HDF5 files for not complete iterations if needed
        if complete_iterations>0:
            clean_hdf5_file_MC_results(hdf5_file_MC_results,worker_id)

        projects.set_current(project)

        #Creating the LCA object --> set fix_dictionaries=False as not useful here?
        lca = DirectSolvingMonteCarloLCA(demand = functional_units[0])
        lca.load_data()
        
        #for LCI physical correlation: water and land transformation        
        corr_ef_code_input=land_transfo_code_input
        corr_ef_code_output=land_transfo_code_output
        corr_twin_ef_input_output=corr_twin_land_transfo
        ratio_rv_dict_per_activity_land_transfo=calculate_ratio_corr_lognormal_for_DB(corr_ef_code_input, 
                                      corr_ef_code_output, 
                                      corr_twin_ef_input_output, 
                                      biosphere_dict, 
                                      bio_params,biosphere_names)

        corr_ef_code_input=water_code_input
        corr_ef_code_output=water_code_output
        corr_twin_ef_input_output=corr_twin_water
        ratio_rv_dict_per_activity_water=calculate_ratio_corr_lognormal_for_DB(corr_ef_code_input, 
                                      corr_ef_code_output, 
                                      corr_twin_ef_input_output, 
                                      biosphere_dict, 
                                      bio_params,biosphere_names)
        
        
        start_iteration = 0
        end_iteration = 0

        #Create and save objects per iteration --> Per iteration: A=0.49MB, B=0.34MB, creation time for both=0.35sec 
        for iteration in range(iterations):

            #Name of the iteration for the storage, starts from 0
            iteration_name=complete_iterations

            print('--Starting job for worker {}, iteration {}, stored as {}, previous complete iteration in {} min'.format(worker_id, iteration,iteration_name,(end_iteration-start_iteration)/60))

            start_iteration = time.time()

            #start_1 = time.time()
            #Creating A and B matrix
            lca.rebuild_technosphere_matrix(lca.tech_rng.next())
            lca.rebuild_biosphere_matrix(lca.bio_rng.next())
            
            #LCI physical correlation
            biosphere_matrix=lca.biosphere_matrix
            lca.biosphere_matrix=generate_corr_biosphere_matrix(ratio_rv_dict_per_activity_land_transfo,biosphere_dict, biosphere_matrix)
            biosphere_matrix=lca.biosphere_matrix
            lca.biosphere_matrix=generate_corr_biosphere_matrix(ratio_rv_dict_per_activity_water,biosphere_dict, biosphere_matrix)
            
            #Saving A and B to HDF5 file
            group_path_techno='/technosphere_matrix/{}'.format(str(iteration_name))
            group_path_bio='/biosphere_matrix/{}'.format(str(iteration_name))
            write_LCA_obj_to_HDF5_file(lca.technosphere_matrix,hdf5_file_MC_results,group_path_techno)
            write_LCA_obj_to_HDF5_file(lca.biosphere_matrix,hdf5_file_MC_results,group_path_bio)
            hdf5_file_MC_results[group_path_techno].attrs['Creation ID']=job_id
            hdf5_file_MC_results[group_path_bio].attrs['Creation ID']=job_id

            #Size_A_MB=(hdf5_file_MC_results[group_path_techno+'/data'].id.get_storage_size()+hdf5_file_MC_results[group_path_techno+'/indptr'].id.get_storage_size()+hdf5_file_MC_results[group_path_techno+'/indices'].id.get_storage_size())/1000000
            #Size_B_MB=(hdf5_file_MC_results[group_path_bio+'/data'].id.get_storage_size()+hdf5_file_MC_results[group_path_bio+'/indptr'].id.get_storage_size()+hdf5_file_MC_results[group_path_bio+'/indices'].id.get_storage_size())/1000000



            #For calculation
            lca.decompose_technosphere()

            #end_1 = time.time()
            #print("Calcul et sauvegarde A et B en {} secondes for iteration {}, and the storage size is A={}MB and B={}MB".format(end_1 - start_1,iteration,Size_A_MB,Size_B_MB)) 

            #Create and save objects per activity/iteration --> Per iteration and activity: supply_array=0.04MB, creation time=0.01sec (except for the first activity=0.5sec)
            for act_index, fu in enumerate(functional_units):

                #start_2 = time.time()

                #Creating UUID for each activity
                actKey = list(fu.keys())[0][1]

                #Create demand_array
                lca.build_demand_array(fu)

                #Create supply_array
                lca.supply_array = lca.solve_linear_system()

                #Save supply_array to HDF5 file
                group_path_supply='/supply_array/{}/{}'.format(actKey,str(iteration_name))
                write_LCA_obj_to_HDF5_file(lca.supply_array,hdf5_file_MC_results,group_path_supply)
                hdf5_file_MC_results[group_path_supply].attrs['Creation ID']=job_id

                #end_2 = time.time()
                #print("Calcul et sauvegarde s en {} secondes for iteration {} and activity {}, and the storage size is s={}MB".format(end_2 - start_2,iteration,actKey,hdf5_file_MC_results[group_path_supply].id.get_storage_size()/1000000)) 

            #Count the number of complete iterations
            complete_iterations=iteration_name+1
            hdf5_file_MC_results.attrs['Number of complete iterations']= complete_iterations

            end_iteration = time.time()


        hdf5_file_MC_results.close()
        
    except (KeyboardInterrupt, SystemExit):
        
        hdf5_file_MC_results.flush()
        hdf5_file_MC_results.close()
        
        print("Process interrupted")
        
        
        
    return;

#TEST OK    
def get_useful_info(collector_functional_unit, hdf5_file_useful_info_per_DB, job_id, activities):

    # Sacrificial LCA to extract relevant information (demand of 1 for all activities)
    # Done on the "collector" functional unit to ensure that all activities and 
    # exchanges are covered in the common dicts (only relevant if some activities 
    # link to other upstream databases
    sacrificial_lca = LCA(collector_functional_unit)
    sacrificial_lca.lci()
    
    #Get data to store 
    product_dict=sacrificial_lca.product_dict
    _product_dict=sacrificial_lca._product_dict
    biosphere_dict=sacrificial_lca.biosphere_dict
    _biosphere_dict=sacrificial_lca._biosphere_dict
    activity_dict=sacrificial_lca.activity_dict
    _activity_dict=sacrificial_lca._activity_dict
    tech_params=sacrificial_lca.tech_params
    bio_params=sacrificial_lca.bio_params
    activities_keys=np.string_([act.key[1] for act in activities])
    #rev_activity_dict, rev_product_dict, rev_bio_dict = sacrificial_lca.reverse_dict()  --> really needed?
    
    #Get metadata for the HDF5 file
    DB_name=activities[0].key[0]
    
    #Write useful information to HDF5 file
    hdf5_file_useful_info_per_DB.attrs['Database name']=DB_name
    hdf5_file_useful_info_per_DB.attrs['Creation ID']=job_id
    hdf5_file_useful_info_per_DB.attrs['Description']='HDF5 file containing all useful information related to the database in order to use dependant Monte Carlo results'
    
    write_LCA_obj_to_HDF5_file(product_dict,hdf5_file_useful_info_per_DB,'/product_dict')
    write_LCA_obj_to_HDF5_file(_product_dict,hdf5_file_useful_info_per_DB,'/_product_dict')
    write_LCA_obj_to_HDF5_file(biosphere_dict,hdf5_file_useful_info_per_DB,'/biosphere_dict')
    write_LCA_obj_to_HDF5_file(_biosphere_dict,hdf5_file_useful_info_per_DB,'/_biosphere_dict')
    write_LCA_obj_to_HDF5_file(activity_dict,hdf5_file_useful_info_per_DB,'/activity_dict')
    write_LCA_obj_to_HDF5_file(_activity_dict,hdf5_file_useful_info_per_DB,'/_activity_dict')
    write_LCA_obj_to_HDF5_file(tech_params,hdf5_file_useful_info_per_DB,'/tech_params')
    write_LCA_obj_to_HDF5_file(bio_params,hdf5_file_useful_info_per_DB,'/bio_params')
    write_LCA_obj_to_HDF5_file(activities_keys,hdf5_file_useful_info_per_DB,'/activities_keys')
    #write_LCA_obj_to_HDF5_file(rev_activity_dict,hdf5_file_useful_info_per_DB,'/rev_activity_dict')
    #write_LCA_obj_to_HDF5_file(rev_product_dict,hdf5_file_useful_info_per_DB,'/rev_product_dict')
    #write_LCA_obj_to_HDF5_file(rev_bio_dict,hdf5_file_useful_info_per_DB,'/rev_bio_dict')

    return None;


#Useful when the code is run from the console to pass arguments to the main function
#@click.command()
#@click.option('--project', default='default', help='Brightway2 project name', type=str)
#@click.option('--database', help='Database name', type=str)
#@click.option('--iterations', default=1000, help='Number of Monte Carlo iterations', type=int)
#@click.option('--cpus', default=mp.cpu_count(), help='Number of used CPU cores', type=int)
#@click.option('--output_dir', help='Output directory path', type=str)



#Create and save useful information during Dependant LCI MC : database objects (_dict, activities, _params, reverse_dict), iteration objects (_sample, i.e. A and B _matrix), act/iteration objects (supply_array)    
def Dependant_LCI_Monte_Carlo_results(project, database, iterations, cpus, activity_category_ID, output_dir):
    
    projects.set_current(project)
    bw2setup()

    #Path the write the results
    BASE_OUTPUT_DIR = output_dir

    #ID to identify who and when was the calculation made
    now = datetime.datetime.now()
    job_id = "{}_{}-{}-{}_{}h{}".format(os.environ['COMPUTERNAME'],now.year, now.month, now.day, now.hour, now.minute)

    #Selection of activities for MC analysis
    db = Database(database)
    activities = [activity for activity in db if activity_category_ID in str(activity['classifications'])]
    #act1=db.get('e929619f245df590fee5d72dc979cdd4')
    #act2=db.get('bdf7116059abfcc6b8b9ade1a641e578')
    #act3=db.get('c8c815c68836adaf964daaa001a638a3')
    #activities = [act1,act2,act3]

    #Create objects to pass the functional units = 1 for each activity
    functional_units = [ {act.key: 1} for act in activities ]
    collector_functional_unit = {k:v for d in functional_units for k, v in d.items()}

    #Create or open the HDF5 file for useful information storage per DB
    path_for_saving=BASE_OUTPUT_DIR
    hdf5_file_name="Useful_info_per_DB.hdf5"
    hdf5_file_useful_info_per_DB_path="{}\\{}".format(path_for_saving,hdf5_file_name)
    
    if os.path.isfile(hdf5_file_useful_info_per_DB_path)==False:

        hdf5_file_useful_info_per_DB=h5py.File(hdf5_file_useful_info_per_DB_path,'a')

        #Create and save all the useful information related to the database only
        get_useful_info(collector_functional_unit, hdf5_file_useful_info_per_DB, job_id, activities)

        hdf5_file_useful_info_per_DB.close()

    #Code to slipt the work between each CPUs of the computer (called workers). The work refers here to the dependant LCI MC for each activity 
    workers = []

    for worker_id in range(cpus):
        #Create or open the HDF5 file for each worker and write metadata
        hdf5_file_name="LCI_Dependant_Monte_Carlo_results_worker"+str(worker_id)+".hdf5"
        hdf5_file_MC_results_path="{}\\{}".format(BASE_OUTPUT_DIR,hdf5_file_name)

        hdf5_file_MC_results=h5py.File(hdf5_file_MC_results_path,'a')

        hdf5_file_MC_results.attrs['Database name']=db.name
        hdf5_file_MC_results.attrs['Worker ID']=worker_id
        hdf5_file_MC_results.attrs['Description']='HDF5 file containing all dependant Monte Carlo results per iteration (A and B matrix) and per activity/iteration (supply_array)'

        hdf5_file_MC_results.close()

        # Create child processes that can work apart from parent process
        child = mp.Process(target=worker_process, args=(projects.current, job_id, worker_id, functional_units, iterations,path_for_saving,collector_functional_unit))
        workers.append(child)
        child.start()
        
    return;
      
#Commande to launch it from the console : python database_wide_monte_carlo_hdf5_storage.py        
        
#Useful when the code is run from the console to execute the main function
if __name__ == '__main__':
    Dependant_LCI_Monte_Carlo_results(project="iw_integration",
                                      database="ecoinvent 3.3 cutoff",
                                      iterations=250,
                                      cpus=4,
                                      activity_category_ID="19a",
                                      output_dir="D:\\Dropbox (MAGI)\\Dossiers professionnels\\Logiciels\\Brightway 2\\Monte Carlo results_with correlation\\Dependant LCI Monte Carlo - sector 19a")