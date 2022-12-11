"""
Preprocess Uniprot Domain Data
"""

import pandas as pd
import re
from bs4 import BeautifulSoup
import urllib.request

"""
CONSTRUCT DICTIONARY
"""


def get_domain_from_uniprot_online(ids):
    # init result dictionary
    all_domain_dict = {}
    # base url
    url = 'https://rest.uniprot.org/uniprotkb/'
    for ID in ids:
        print('Searching for ' + ID + ' in UniProt database...')
        curr_domains = []
        # read url get xml
        url_response = urllib.request.urlopen(url + ID + '.xml')
        xml_content = url_response.read()
        content = BeautifulSoup(xml_content, features='xml')
        features = content.find_all("feature")
        for feat in features:
            if feat.attrs['type'] == 'domain':
                domain_name = feat.attrs['description']
                start = eval(feat.begin.attrs['position'])
                end = eval(feat.end.attrs['position'])
                curr_domains.append([domain_name, start, end])
        # add domain to final result
        if curr_domains:
            all_domain_dict[ID] = curr_domains
    return all_domain_dict


def get_domain_from_tsv(domain_file_path='uniprot_reviewed_proteins_with_domains.tsv'):
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
        curr_domain_arr = []
        # curr_domain_dict = {}
        prot_id = domains_nonan.iloc[row]['Entry']
        domain_arr = domains_nonan.iloc[row]['Domain [FT]'].split('DOMAIN ')[1:]
        # for each of the domains of this protein
        for domain in domain_arr:
            domain_split_arr = domain.split('; /')
            # extract start & end positions, converting to 0-index
            start_end = [eval(num)-1 for num in re.findall(r'\d+', domain_split_arr[0])]
            # if this entry has valid start & end positions
            if len(start_end) == 2:
                start, end = [eval(num)-1 for num in re.findall(r'\d+', domain_split_arr[0])]
            else:
                continue
            # extract domain name
            domain_name = re.findall(r'"([^"]*)"', domain_split_arr[1])[0]
            # record domain name, start, end
            # if domain_name not in curr_domain_dict:
            #     curr_domain_dict[domain_name] = [start_end]
            # else:
            #     curr_domain_dict[domain_name].append(start_end)
            curr_domain_arr.append([domain_name, start, end])
        all_domain_dict[prot_id] = curr_domain_arr
    return all_domain_dict


def sequence_to_domain_structure(prot_id, seq, all_domains):
    """
    :param seq: str sequence to extract domain from
    :param prot_id: str UniProt ID of sequence to extract domain from
    :param all_domains: dict preconstructed dictionary from UniProt
    :return: list of domain & linker names & start end locations of the sequence (0-index), structure list
    """
    # if this id does not contain known domain, we treat entire sequence as a big linker
    if prot_id not in all_domains:
        # print(prot_id + ' not found in UniProt Domain Database! Proceeding as _unknown_')
        # return {'linker1_': [0, len(seq)-1], 'structure_list_': ['linker1_']}
        return [['linker1_', 0, len(seq)-1]], ['linker1_']
    # get protein length
    length = len(seq)
    # if this id contains known domain(s), we find domain & linker positions
    entry = all_domains[prot_id]
    # init result dictionary
    # res = {}
    res = []
    structure_list = []
    # record linker positions
    linker_cnt = 0
    previous_end = -1
    # iterate through domains in order except length_, which is not domain
    for domain in entry:
        domain_name = domain[0]
        curr_start = domain[1]
        curr_end = domain[2]
        if previous_end != curr_start - 1:
            linker_cnt += 1
            # add linker & domain name, start, end to result
            res.append(['linker' + str(linker_cnt) + '_', previous_end + 1, curr_start - 1])  # if 1 base long, 2 nums same
            res.append(domain)
            previous_end = curr_end
            # add previous linker to structure list
            structure_list.append('linker' + str(linker_cnt) + '_')
            # add current domain to structure list
            structure_list.append(domain_name)
    # check if linker exists after last domain (C-terminus segment)
    if previous_end < length - 1:
        linker_cnt += 1
        res.append(['linker' + str(linker_cnt) + '_', previous_end + 1, length - 1])
        structure_list.append('linker' + str(linker_cnt) + '_')
    # res['structure_list_'] = structure_list
    return res, structure_list


"""
TEST CASE
"""
# all_domains = parse_panda_to_dict(domain_file_path='uniprot_crk.tsv')
# print(sequence_to_domain_structure('A0A6J3HBX8', 'MSSARFDSSDRSAWYMGPVSRQEAQNRLQGQRHGMFLVRDSSTCPGDYVLSVSENSRVSHYIINSLP'
#                                                  'NRRFKIGDQEFDHLPALLEFYKIHYLDTTTLIEPAPRYPSPPMGSVSAPSL'
#                                                  'PTAEENLEYVRTLYDFPGNDAEDLPFKKGEILVIIEKPEEQWWSARNKDGRVGMIPVPYVEKLVRSSPH'
#                                                  'GKHGNRNSNSYGIPEPAHAYAQPQTTTPIPAVSGSPGAAITPLPSTQNGPVFAKAIQKRVPCA'
#                                                  'YDKTALALEVGDIVKVTRMNINGQWEGEVNGRKGLFPFTHVKIFDPQNPDENE', all_domains))
#
# print(sequence_to_domain_structure('A0A2K5JNX9', 'MSSARFDSSDRSAWYMGPVSRQEAQTRLQGQRHGMFLVRDSSTCPGDYVLSVSENSRVSH'
#                                                  'YIINSLPNRRFKIGDQEFDHLPALLEFYKIHYLDTTTLIEPAPRYPSPPMGSVSAPNLPT'
#                                                  'AEDNLEYVRTLYDFPGNDAEDLPFKKGEILVIIEKPEEQWWSARNKDGRVGMIPVPYVEKLV'
#                                                  'RSSPHGKHGNRNSNSYGIPEPAHAYAQPQTTTPLPAVSGSPGAAITPLPSTQNGPVFAKAI'
#                                                  'QKRVPCAYDKTALALEVGDIVKVTRMNINGQWEGEVNGRKGLFPFTHVKIFDPQNPDENE',
#                                    all_domains))



