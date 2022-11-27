"""
Preprocess Uniprot Domain Data
"""

import pandas as pd
import re

"""
CONSTRUCT DICTIONARY
"""


def parse_panda_to_dict(domain_file_path='ICA/BMI3_domain_aware_aligner/uniprot_reviewed_proteins_with_domains.tsv'):
    """
    :param domain_file_path: str path of domain file
    :param domains_nonan: DataFrame: raw uniprot pandas
    :return: dict: nested dictionary of 0-index domain positions for UniProt IDs
    """
    # read in domains
    domains = pd.read_csv(domain_file_path, sep='\t')
    domains_nonan = domains.dropna()
    # init result dictionary
    all_domain_dict = {}
    # for each of the entries
    for row in range(domains_nonan.shape[0]):
        curr_domain_dict = {}
        prot_id = domains_nonan.iloc[row]['Entry']
        domain_arr = domains_nonan.iloc[row]['Domain [FT]'].split('DOMAIN ')[1:]
        length = int(domains_nonan.iloc[row]['Length'])
        # add length info to curr_domain_dict
        curr_domain_dict['length_'] = length
        # for each of the domains of this protein
        for domain in domain_arr:
            domain_split_arr = domain.split('; /')
            # extract start & end positions, converting to 0-index
            start_end = [eval(num)-1 for num in re.findall(r'\d+', domain_split_arr[0])]
            # extract domain name
            domain_name = re.findall(r'"([^"]*)"', domain_split_arr[1])[0]
            curr_domain_dict[domain_name] = start_end
        all_domain_dict[prot_id] = curr_domain_dict
    return all_domain_dict


def sequence_to_domain_structure(prot_id, seq, all_domains):
    """
    :param seq: str sequence to extract domain from
    :param prot_id: str UniProt ID of sequence to extract domain from
    :param all_domains: dict preconstructed dictionary from UniProt
    :return: dict domain & linker locations of the sequence (0-index), structure list
    """
    # if this id does not contain known domain, we treat entire sequence as a big linker
    if prot_id not in all_domains:
        print(prot_id + ' not found in UniProt Domain Database! Proceeding as Linker')
        return {'linker1_': [0, len(seq)-1],
                'structure_list_': ['linker1_']}
    # get protein length
    length = all_domains[prot_id]['length_']
    # if this id contains known domain(s), we find domain & linker positions
    entry = all_domains[prot_id]
    res = entry.copy()
    structure_list = []
    # record linker positions
    linker_cnt = 0
    previous_end = -1
    # iterate through domains in order except length_, which is not domain
    for domain in entry:
        if domain != 'length_':
            # get start & end position of current domain
            curr_start = entry[domain][0]
            curr_end = entry[domain][1]
            # if previous end is not one nucleotide before current start, this means a linker exists
            if previous_end != curr_start - 1:
                linker_cnt += 1
                res['linker' + str(linker_cnt) + '_'] = [previous_end + 1, curr_start - 1]  # if 1 base long, 2 nums same
                previous_end = curr_end
                # add previous linker to structure list
                structure_list.append('linker' + str(linker_cnt) + '_')
                # add current domain to structure list
                structure_list.append(domain)
    # check if linker exists after last domain (C-terminus segment)
    if previous_end < length - 1:
        linker_cnt += 1
        res['linker' + str(linker_cnt) + '_'] = [previous_end + 1, length - 1]
        structure_list.append('linker' + str(linker_cnt) + '_')
    res['structure_list_'] = structure_list
    return res


"""
TEST CASES
"""
all_domains = parse_panda_to_dict()
print(sequence_to_domain_structure('A0A024SH76', '', all_domains)) # dont do this, this empty string is used only if id not in all_domains



