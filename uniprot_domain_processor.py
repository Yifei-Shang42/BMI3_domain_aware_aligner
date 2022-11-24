"""
Preprocess Uniprot Domain Data
"""

import pandas as pd
import re

"""
INPUT
"""

domains = pd.read_csv('ICA/BMI3_domain_aware_aligner/uniprot_reviewed_proteins_with_domains.tsv', sep='\t')
domains_nonan = domains.dropna()

"""
CONSTRUCT DICTIONARY
"""


def parse_panda_to_dict(domains_nonan):
    """
    :param domains_nonan: DataFrame: raw uniprot pandas
    :return: dict: nested dictionary
    """
    all_domain_dict = {}
    # for each of the entries
    for row in range(domains_nonan.shape[0]):
        curr_domain_dict = {}
        prot_id = domains_nonan.iloc[row]['Entry']
        domain_arr = domains_nonan.iloc[row]['Domain [FT]'].split('DOMAIN ')[1:]
        # for each of the domains of this protein
        for domain in domain_arr:
            domain_split_arr = domain.split('; /')
            # extract start & end positions
            start_end = re.findall(r'\d+', domain_split_arr[0])
            # extract domain name
            domain_name = re.findall(r'"([^"]*)"', domain_split_arr[1])[0]
            curr_domain_dict[domain_name] = start_end
        all_domain_dict[prot_id] = curr_domain_dict
    return all_domain_dict
all_domains = parse_panda_to_dict(domains_nonan)