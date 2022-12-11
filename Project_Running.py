import argparse
import json
import pandas as pd
import os
from multi_alignment import *
from uniprot_domain_processor import  *

def readfile(i:str):
  with open(i) as fl:
    content= {}
    uniprot_inf = []
    entry_id = fl.readline().strip().split('|')[1]
    seq = ''
    for i in fl:
      if '>' in i:
        uniprot_inf.append(i.strip().split('|'))
        content[entry_id] = seq
        entry_id = i.strip().split('|')[1]
        seq = ''
      else:
        seq += i.strip().replace('-','')
    content[entry_id] = seq
  return content, uniprot_inf

def readReferenceFile(r:str):
  with open(r) as reference:
    reference_tsv = pd.read_csv(
      reference,
      sep='\t',
      header=0
    )
    return reference_tsv

def cut_text(text,lenth):
 textArr = re.findall('.{'+str(lenth)+'}', text)
 textArr.append(text[(len(textArr)*lenth):])
 return textArr

def OutPut(data,file,uniprot_inf:list):
    for item in data:
        second_dim = data[item]
        IDs = second_dim['ids']
        Category_alignments = second_dim['category_alignment']
        for i in range(0,len(IDs)):
            for j in uniprot_inf:
                if j[1] == IDs[i]:
                    file.write(j[0] + '|' + IDs[i] + '|' + list(data.keys())[0] + '|' + j[2] + '\n')
                    current_alignment = cut_text(Category_alignments[i],72)
                    for seq in current_alignment:
                        file.write(seq + '\n')

parser = argparse.ArgumentParser(description='ArgparseTry')
parser.add_argument('-i', required=True, type=str, help="the path to readin fasta file")
parser.add_argument('-o', required=False, nargs='?', type=str, default=os.path.abspath('.'), help="the path to show the result txt file")
parser.add_argument('-r', required=False, nargs='?', type=str, default='', help="the path to readin uniprot reference file")
parser.add_argument('-ref_mode', required=False, nargs='?', type=str, default='', help="Uniprot Online Database Reference")
args = parser.parse_args()

if args.r != '':
  reference_tsv = readReferenceFile(args.r)

assert args.ref_mode in ['tsv', 'uniprot', 'ncbi'], "Invalid mode! ref_mode must be in ['tsv', 'uniprot', 'ncbi']"

if args.ref_mode == 'tsv':
  assert args.r != '', '-r must be specified under tsv mode'


if __name__ == '__main__':
    mode = 'online'
    # parsing data & uniprot domain info
    data = readfile(args.i)
    if args.ref_mode == 'tsv':
        all_domains = parse_panda_to_dict(args.r)
    elif args.ref_mode == 'uniprot':
        print('Retrieving UniProt Domain Annotation online...')
        all_domains = get_domain_dict_online(list(data[0].keys()))
    # main algorithm
    domain_alignment_result = domain_aware_greedy_MSA(all_domains, data[0])
    # extracting alignments from result
    print('Outputing results: ')

    if args.o[-1] == '/':
        file = open(args.o + 'result.fasta', 'w')
    else:
        file = open(args.o + '/result.fasta', 'w')

    OutPut(domain_alignment_result,file,data[1])

    for structure in domain_alignment_result:
        print('-------------------------------------------------------')
        print('Alignment result for '+str(len(domain_alignment_result[structure]['seqs']))+' sequences of '+structure+' structure: ')
        for alignment in domain_alignment_result[structure]['category_alignment']:
            print(alignment)