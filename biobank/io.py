import sys
import os
import re

from biobank import meshtools
from meshtools.polydata.io import read_PolyData 
import biobank.numpyIO as numpyIO


def generate_structures():
    """
    Dict of structures to regex
    """
    
    structures = {
        'br_stem': 'BrStem',
        'l_accu': 'L_Accu',
        'l_amyg': 'L_Amyg',
        'l_caud': 'L_Caud',
        'l_hipp': 'L_Hipp',
        'l_pall': 'L_Pall',
        'l_puta': 'L_Puta',
        'l_thal': 'L_Thal',
        'r_accu': 'R_Accu',
        'r_amyg': 'R_Amyg',
        'r_caud': 'R_Caud',
        'r_hipp': 'R_Hipp',
        'r_pall': 'R_Pall',
        'r_puta': 'R_Puta',
        'r_thal': 'R_Thal'
    }
    
    return structures


def generate_subject_ids(root_dir, structures):
    """
    Counterpart to vtkIO.generate_dataset(root_dir, structures).
    
    Outputs a list of subject_ids (strings) that matches the list output by the above function.
    
    Useful to save some output files after processing the input data.    
    
    !!! This only works if root_dir is the/path/to/the/folder/immediately/before/subject/specific/folders !!!
    (immediately under folder shapes there are subject specific -- numeric format -- folders for me)
    """
    
    result = []
    
    for root, dirs, files in os.walk(root_dir):
        rel_dir = os.path.relpath(root, root_dir)
        
        subject_output = {}
        for name in files:
            for key in structures:
                if re.search(structures[key], name):
                    subject_output[key] = os.path.join(rel_dir, name)
                    break

        if subject_output:
            result.append(rel_dir)           
            
    return result     

def generate_data_ids(subject_dataset, subject_ids):
    """
    subject_dataset: list of (structure, value)-dictionaries, 1 per subject; 
                    e.g. the output of generate_dataset_filenames(root_dir, structures)
    subject_ids: list of strings (1 per subject); the output of generate_subject_ids(root_dir, structures)
    
    Returns: A list of (subject_id, structure) tuples, as many as the total number of entries in all subject dictionaries
    """
    
    result = []
    
    for i in range(len(subject_dataset)):
        data = subject_dataset[i]
        subject_id = subject_ids[i]
        
        for key in data.keys():
            result.append( (subject_id, key) )
        
    return result

def read_subject_polydatas(dataset_filenames, root_dir):
    """
    dataset_filename: the output of generate_dataset_filenames(root_dir, structures)
    
    returns a list of (structure, vtkPolyData)-dictionaries, 1 per subject.
    """
    
    result = []
    
    for subject in dataset_filenames:
        subject_structures = {}
        for key in subject.keys():
            subject_structures[key] = read_PolyData(os.path.join(root_dir, subject[key]))
        
        result.append(subject_structures)
        
    return result

def read_subject_csr_maps(dataset_filenames, root_dir):
    """
    dataset_filename: the output of generate_dataset_filenames(root_dir, structures)
    
    returns a list of (structure, PolyData)-dictionaries, 1 per subject.
    """
    
    result = []
    
    for subject in dataset_filenames:
        subject_structures = {}
        for key in subject.keys():
            subject_structures[key] = numpyIO.load_sparse_csr(os.path.join(root_dir, subject[key]))
        
        result.append(subject_structures)
        
    return result

def generate_data(subject_dataset):
    """
    Reformats subject_dataset, the output of read_subject_polydatas.
    Output: List of PolyData, as many as the total number of entries in all subject dictionaries
    
    See generate_dat_ids to find out what maps to which subject/structure if required.
    """
    
    result = []
    
    for data in subject_dataset:
        result.extend([ data[key] for key in data ])
        
    return result