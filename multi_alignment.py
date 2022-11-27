"""
BMI3 Group5 Project4: Domain-aware aligner
"""


import numpy as np
from math import inf
import blosum as bl
from ICA.BMI3_domain_aware_aligner.uniprot_domain_processor import *
mat = bl.BLOSUM(62)


"""
PARSE INPUT
"""


def parse_txt_to_dict(path):
    """
    :param path: str path of a txt file of fasta format
    :return: dict {id: seq, ...}
    """
    with open(path, 'r') as infile:
        lines = infile.readlines()
    fasta_dict = {}
    curr_seq = ''
    curr_id = None
    for row in lines:
        if row[0] == '>':
            # record, except for beginning
            if curr_id is not None:
                fasta_dict[curr_id] = curr_seq
            # next seq
            curr_seq = ''
            curr_id = row.split('|')[1]
        else:
            curr_seq += row.strip().replace('-', '')
    # add last
    fasta_dict[curr_id] = curr_seq
    return fasta_dict


"""
CORE ALGORITHM: PAIR-WISE GLOBAL ALIGNMENT WITH AFFINE GAP PENALTY
"""


def affine_gap_penalties_pair_wise_alignment(seq1, seq2=None, sigma=11, epsilon=1, mode='pairwise', alignment=None):
    """
    :param mode: str select between 'pairwise' & 'profile' modes
    :param seq1: str input protein sequence 1
    :param seq2: str input protein sequence 2
    :param sigma: int penalty for opening a gap
    :param epsilon: int penalty for extending a gap
    :param alignment: list of alignment strings
    :return: tuple (align1: str alignment for seq1, align2: str alignment for seq2, score: float best alignment score)
    """
    def gen_backtrack(seq1, seq2, sigma=11, epsilon=1, mode='pairwise', profile=None):
        """
        :param profile: 2d nested list of profile matrix of previous alignment
        :param mode: str select between 'pairwise' & 'profile' modes
        :param seq1: str input protein sequence 1
        :param seq2: str input protein sequence 2
        :param sigma: int penalty for opening a gap
        :param epsilon: int penalty for extending a gap
        :return: tuple (backtracks: list [3 np.array backtracks for gap1, match, and gap2], score: float best alignment score)
        """
        def weighted_score_with_profile(base, profile_column, penalty):
            """
            :param base: str the base in sequence 1
            :param profile_column: list of float the column of alignment profile matrix at corresponding position
            :param penalty: dictionary penalty matrix
            :return: weighted_score: float score of weighted score based on profile
            """
            amino_arr = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                     'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
            weighted_score = sum([])
            for row in range(len(profile_column)):
                weighted_score += penalty[base + amino_arr[row]] * profile_column[row]
            return weighted_score
        # init penalty
        penalty = bl.BLOSUM(62)
        len1 = len(seq1)
        if mode == 'pairwise':
            len2 = len(seq2)
        elif mode == 'profile':
            assert profile is not None, 'profile parameter required under profile mode!'
            len2 = len(profile[0])
        else:
            assert False, 'Invalid mode! Please select from "pairwise" and "profile"!'
        # init scores
        match_scores = -inf * np.ones((len1+1, len2+1))
        gap1_scores = -inf * np.ones((len1+1, len2+1))
        gap2_scores = -inf * np.ones((len1+1, len2+1))
        # init backtracks
        # what are backtracks:
        # they are matrices of the same dimension as score matrices, used to record the best previous position
        match_back = [['' for _ in range(len2+1)] for _ in range(len1+1)]
        gap1_back = [['' for _ in range(len2+1)] for _ in range(len1+1)]
        gap2_back = [['' for _ in range(len2+1)] for _ in range(len1+1)]
        # at start position, all scores are 0
        gap1_scores[0, 0] = 0
        gap2_scores[0, 0] = 0
        match_scores[0, 0] = 0
        # init first gap openings
        gap1_scores[1, 0] = -sigma
        gap2_scores[0, 1] = -sigma
        match_scores[1, 0] = -sigma
        match_scores[0, 1] = -sigma
        # init lateral gap extensions: technically, we have 3 score matrices, we first calculate along all edges
        for pos1 in range(2, len1 + 1):
            gap1_scores[pos1, 0] = gap1_scores[pos1 - 1, 0] - epsilon
            match_scores[pos1, 0] = gap1_scores[pos1, 0]
        for pos2 in range(2, len2 + 1):
            gap2_scores[0, pos2] = gap2_scores[0, pos2 - 1] - epsilon
            match_scores[0, pos2] = gap2_scores[0, pos2]
        # start dynamic traversing: technically, we start to dynamically fill in the 3 matrices
        # unlike normal alignment, for affine gap alignment, we need 3 matrices,
        # so for each point, we consider 3 states: in gap1, match, in gap2
        for pos1 in range(1, len1 + 1):
            for pos2 in range(1, len2 + 1):
                # First, consider we are now in gap1
                # to best get to gap 1, do we open new gap? or extend existing gap?
                # if we extend, we move horizontally in gap1 state
                # if we open, we move diagonally from match state to gap1 state
                gap1_extend_score = gap1_scores[pos1 - 1, pos2] - epsilon
                gap1_open_score = match_scores[pos1 - 1, pos2] - sigma
                gap1_best_score = np.max([gap1_open_score, gap1_extend_score])
                gap1_scores[pos1, pos2] = gap1_best_score
                # if we open gap:
                if gap1_best_score == gap1_open_score:
                    # we record at gap1 backtrack as from match
                    gap1_back[pos1][pos2] = 'match'
                else:
                    # if extend
                    gap1_back[pos1][pos2] = 'gap1'
                # Then, consider we are now in gap2
                # to best get to gap 2, do we open new gap? or extend existing gap?
                # if we extend, we move horizontally in gap2 state
                # if we open, we move diagonally from match state to gap2 state
                gap2_extend_score = gap2_scores[pos1, pos2 - 1] - epsilon
                gap2_open_score = match_scores[pos1, pos2 - 1] - sigma
                gap2_best_score = np.max([gap2_open_score, gap2_extend_score])
                gap2_scores[pos1, pos2] = gap2_best_score
                # if we should open gap:
                if gap2_best_score == gap2_open_score:
                    # we record at gap1 backtrack as from match
                    gap2_back[pos1][pos2] = 'match'
                else:
                    gap2_back[pos1][pos2] = 'gap2'
                # Last, consider we are now in match state
                # do we continue with gaps / do a match?
                # match_match_score = match_scores[pos1 - 1, pos2 - 1] + penalty_matrix[aa_idx[seq1[pos1-1]]][aa_idx[seq2[pos2-1]]]
                if mode == 'pairwise':
                    match_match_score = match_scores[pos1 - 1, pos2 - 1] + penalty[seq1[pos1-1] + seq2[pos2-1]]
                elif mode == 'profile':
                    # get the column to calculate weighted score from profile
                    profile_col = [row[pos2-1] for row in profile]
                    match_match_score = match_scores[pos1 - 1, pos2 - 1] + weighted_score_with_profile(seq1[pos1-1], profile_col, penalty)
                best_score = np.max([gap1_best_score, match_match_score, gap2_best_score])
                match_scores[pos1, pos2] = best_score
                if best_score == match_match_score:
                    match_back[pos1][pos2] = 'match'
                elif best_score == gap2_best_score:
                    match_back[pos1][pos2] = 'gap2'
                elif best_score == gap1_best_score:
                    match_back[pos1][pos2] = 'gap1'
        return [gap1_back, match_back, gap2_back], match_scores[len1, len2]

    def gen_alignment_from_backtrack(backtracks, seq1, seq2, mode='pairwise', profile=None):
        """
        :param profile: 2d nested list of profile matrix of previous alignment
        :param mode: str select between 'pairwise' & 'profile' modes
        :param backtracks: list [3 np.array backtracks for gap1, match, and gap2]
        :param seq1: str input protein sequence 1
        :param seq2: str input protein sequence 2
        :return: tuple (align1: str alignment for seq1, align2: str alignment for seq2)
        """
        # start at the end of backtrack
        pos1 = len(seq1)
        if mode == 'pairwise':
            pos2 = len(seq2)
        elif mode == 'profile':
            assert profile is not None, 'profile parameter required under profile mode!'
            pos2 = len(profile[0])
            # if we align to profile, seq2 will be a dummy sequence here just to record the gaps
            seq2 = ''.join(['X' for _ in range(len(profile[0]))])
        else:
            assert False, 'Invalid mode! Please select from "pairwise" and "profile"!'
        # extract backtracts
        gap1_back, match_back, gap2_back = backtracks
        # init alignment strings
        align1 = ''
        align2 = ''
        state = 'match'  # match
        # while not finished, we go back one step at a time
        while pos1 > 0 or pos2 > 0:
            # if we now in gap1
            if state == 'gap1':
                # if this is a gap opening, we go back to match state
                if gap1_back[pos1][pos2] == 'match':
                    state = 'match'
                # we move back one step in seq1, but not seq2 because we are in gap1
                pos1 -= 1
                align1 += seq1[pos1]
                align2 += '-'
            # if we now in gap2
            elif state == 'gap2':
                # if this is a gap2 opening, we go back to match state
                if gap2_back[pos1][pos2] == 'match':
                    state = 'match'
                # we move back one step in seq2, but not seq1 because we are in gap2
                pos2 -= 1
                align1 += '-'
                align2 += seq2[pos2]
            # if we now in match state
            elif state == 'match':  # (this can be changed to else but less clear)
                # what did we do last time? did we come from match / gap1 closing / gap2 closing?
                prev_state = match_back[pos1][pos2]
                # if we came from a match, we go back one step in BOTH seq1 and seq2
                if prev_state == 'match':
                    pos1 -= 1
                    pos2 -= 1
                    align1 += seq1[pos1]
                    align2 += seq2[pos2]
                # if we came from either gap, we go one step back to gap1 / gap2 state
                elif prev_state in ['gap1', 'gap2']:  # (this can be changed to else but less clear)
                    state = prev_state
        # when we are at the start, we return results
        return align1, align2

    def alignment_to_profile(alignment):
        """
        :param alignment: list of alignment strings
        :return: matrix: 2d-nested list of profile matrix
        """
        matrix = []
        for base in ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                     'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']:
            mat = []
            for pos in range(len(alignment[0])):
                col = [row[pos] for row in alignment]
                mat.append(col.count(base) / len(alignment))
            matrix.append(mat)
        return matrix
    # initialize results
    align1 = align2 = score = None
    assert mode in ['pairwise', 'profile'], \
        'Invalid Mode: available modes: [pairwise, profile]'
    if mode == 'pairwise':
        # normal pairwise procedure
        backtracks, score = gen_backtrack(seq1, seq2, sigma, epsilon, mode)
        align1, align2 = gen_alignment_from_backtrack(backtracks, seq1, seq2, mode)
    if mode == 'profile':
        assert alignment is not None, 'alignment parameter required under "profile" mode!'
        # first generate profile of the alignment
        profile = alignment_to_profile(alignment)
        backtracks, score = gen_backtrack(seq1, seq2, sigma, epsilon, mode, profile)
        align1, align2 = gen_alignment_from_backtrack(backtracks, seq1, seq2, mode, profile)
    return align1[::-1], align2[::-1], score


"""
(DEPRECATED) GREEDY ALGO FOR MSA BASED ON PAIR-WISE GLOBAL ALIGNMENT: !NOT CONSIDERING DOMAIN!
"""


def greedy_multiple_alignment_with_affine_gap_penalties(seqs, sigma=11, epsilon=1):
    """
    :param seqs: list of str raw strings for MSA input
    :param sigma: int penalty for opening a gap
    :param epsilon: int penalty for extending a gap
    :return: list of str final MSA results
    """
    def init_best_pair_alignment(seqs, sigma=11, epsilon=1):
        """
        :param seqs: list collection of multiple input protein sequences
        :param sigma: int penalty for opening a gap
        :param epsilon: int penalty for extending a gap
        :return: tuple (best_pair: tuple best pair of sequences, best_align: list best pair-wise alignment result)
        """
        best_pair = None
        best_score = -inf
        best_align = None
        # iterate through all seqs to find the best pair-wise alignment
        for pos1 in range(len(seqs) - 1):
            for pos2 in range(pos1 + 1, len(seqs)):
                seq1 = seqs[pos1]
                seq2 = seqs[pos2]
                align1, align2, curr_score = affine_gap_penalties_pair_wise_alignment(seq1, seq2, sigma, epsilon, mode='pairwise')
                if curr_score > best_score:
                    best_score = curr_score
                    best_pair = (seq1, seq2)
                    best_align = [align1, align2]
        return best_pair, best_align

    def add_one_seq_to_alignment(unaligned_seqs, alignment, sigma=11, epsilon=1):
        """
        :param unaligned_seqs: list of str unaligned sequences :param alignment: list of str previous alignment sequences
        :param sigma: int penalty for opening a gap
        :param epsilon: int penalty for extending a gap
        :return: tuple (unaligned_seqs: list of unaligned sequences excluding the newly added seq,
        alignment_new: list of new alignment sequences)
        """
        def update_alignment(alignment, align2_dummy):
            """
            :param alignment: list of old alignment strings
            :param align2_dummy: str align2 generated recording information about new gaps to be added in old alignments
            :return: alignment_new: list of updated alignment strings with new gaps added
            """
            alignment_new = []
            for align in alignment:
                align_new = ''
                align_pos = 0
                for pos in range(len(align2_dummy)):
                    if align2_dummy[pos] == '-':
                        align_new += '-'
                    else:
                        align_new += align[align_pos]
                        align_pos += 1
                alignment_new.append(align_new)
            return alignment_new
        # find the best seq to add to alignment
        best_align1 = None
        best_align2 = None
        best_score = -inf
        best_raw_seq = None
        for seq in unaligned_seqs:
            # for seq2 required in affine_gap_penalties_pair_wise_alignment we use a dummy sequence
            align1, align2_dummy, curr_score = affine_gap_penalties_pair_wise_alignment(seq, ''.join('X' for _ in range(len(alignment[0]))),
                                                                                        sigma, epsilon, mode='profile', alignment=alignment)
            if curr_score > best_score:
                best_align1 = align1
                best_align2 = align2_dummy
                best_score = curr_score
                best_raw_seq = seq
        # update the existing alignment based on new gap information recorded in align2_dummy
        alignment_new = update_alignment(alignment, best_align2)
        alignment_new.append(best_align1)
        unaligned_seqs.remove(best_raw_seq)
        return unaligned_seqs, alignment_new
    # first, we find the pair with highest score
    best_pair, curr_align = init_best_pair_alignment(seqs, sigma, epsilon)
    # record aligned seqs
    unaligned_seqs = [seq_ for seq_ in seqs if seq_ not in best_pair]
    # while not all aligned
    while unaligned_seqs:
        unaligned_seqs, curr_align = add_one_seq_to_alignment(unaligned_seqs, curr_align, sigma, epsilon)
    return curr_align


"""
MAIN ALGORITHM: GREEDY MSA DOMAIN AWARE ALIGNER
"""


def domain_aware_greedy_MSA(all_domains, id_seq_dict, sigma=11, epsilon=1):
    """
    :param all_domains: dict preconstructed dictionary from UniProt
    :param id_seq_dict: dict {id: seq, ...}
    :param sigma: int penalty for opening a gap
    :param epsilon: int penalty for extending a gap
    :return: dict {structure_identifier: {}, ...}
    """
    def categorize_seqs_by_domain_info(all_domains, ids, seqs):
        """
        :param all_domains: dict preconstructed dictionary from UniProt
        :param ids: list of str UniProt IDs of seqs
        :param seqs: list of str raw strings for MSA input
        :return: dict {strucure_identifier: {'seqs': [sequences of this structure], 'ids': [ids of sequences]}, ...}
        """
        # categorize seqs by domain structures
        categories = {}
        for i in range(len(ids)):
            curr_id = ids[i]
            curr_seq = seqs[i]
            curr_domain_info = sequence_to_domain_structure(curr_id, curr_seq, all_domains)
            domain_structure_list = curr_domain_info['structure_list_']
            domain_structure_list_no_linkers = [name for name in domain_structure_list if name[-1] != '_']
            # unique identifier of a sequence is all domain names separated by '---' Example: 'KH 1---DH---VHS' 3 domains
            if not domain_structure_list_no_linkers:
                structure_identifier = '_unknown_'
            else:
                structure_identifier = '---'.join(domain_structure_list_no_linkers)
            if structure_identifier not in categories:
                categories[structure_identifier] = {'seqs': [curr_seq], 'ids': [curr_id]}
            else:
                categories[structure_identifier]['seqs'].append(curr_seq)
                categories[structure_identifier]['ids'].append(curr_id)
        return categories

    def split_seq_by_structure_dict(structure_dict, seq):
        """
        :param structure_dict: dict generated by sequence_to_domain_structure function
        :param seq: sequence to be split
        :return: list of split domains & linkers (linker could be empty string)
        """
        # init res
        split_seq_list = []
        # iterate through each domain & linker
        structure_list_ = structure_dict['structure_list_']
        for structure in structure_list_:
            # start & end are 0-index, remember to use end+1 for indexing
            start, end = structure_dict[structure]
            split_seq_list.append(seq[start: end + 1])
        return split_seq_list

    def init_best_pair_alignment_domain_based(ids, seqs, sigma, epsilon, all_domains):
        """
        :param ids: list of str UniProt IDs of seqs
        :param all_domains: dict pre-constructed dictionary from UniProt
        :param seqs: list collection of multiple input protein sequences
        :param sigma: int penalty for opening a gap
        :param epsilon: int penalty for extending a gap
        :return: tuple (best_pair: tuple best pair of sequences, best_align: list best pair-wise alignment result)
        """
        # if a single category, we return itself
        if len(seqs) == 1:
            return -1, seqs
        # if not, we start to look for the best pair
        best_pair = None
        best_id_pair = None
        best_score = -inf
        best_align = None
        # iterate through all seqs to find the best pair-wise alignment
        for pos1 in range(len(seqs) - 1):
            for pos2 in range(pos1 + 1, len(seqs)):
                seq1 = seqs[pos1]
                seq2 = seqs[pos2]
                # use id to extract domain structure list
                id1 = ids[pos1]
                id2 = ids[pos2]
                ##########
                # print('Checking if '+id1+' and '+id2+' are best init alignment')  # testing only, output too long...
                ##########
                seq1_structure_dict = sequence_to_domain_structure(id1, seq1, all_domains)
                seq2_structure_dict = sequence_to_domain_structure(id2, seq2, all_domains)
                # split both sequences to domains & linkers, because same category, should be split in exact same way
                seq1_split = split_seq_by_structure_dict(seq1_structure_dict, seq1)
                seq2_split = split_seq_by_structure_dict(seq2_structure_dict, seq2)
                ##########
                assert len(seq1_split) == len(seq2_split), 'seq1 & seq2 are of different domain structures'
                ##########
                # iterate through domains & linkers, produce fragmented alignment in a list
                final_align1 = []
                final_align2 = []
                # init final score, we will add curr_score of each fragment in loop below
                final_score = 0
                for i in range(len(seq1_split)):
                    fragment1 = seq1_split[i]
                    fragment2 = seq2_split[i]
                    frag_align1, frag_align2, curr_score = affine_gap_penalties_pair_wise_alignment(fragment1, fragment2, sigma,
                                                                                                    epsilon, mode='pairwise')
                    final_align1.append(frag_align1)
                    final_align2.append(frag_align2)
                    final_score += curr_score
                if final_score > best_score:
                    best_score = curr_score
                    best_id_pair = (id1, id2)
                    best_pair = (seq1_split, seq2_split)
                    best_align = [final_align1, final_align2]
        ###################
        print('Best initial pair found: '+best_id_pair[0]+', '+best_id_pair[1])
        print(best_id_pair[0]+' and '+best_id_pair[1]+' has been aligned! '+str(len(ids)-2)+' proteins to be aligned...')
        print('-------------------------------------------------------')
        ###################
        return best_pair, best_align

    def add_one_seq_to_alignment_domain_based(unaligned_ids, unaligned_seqs, alignment, sigma, epsilon, all_domains):
        """
        :param unaligned_seqs: list of str unaligned sequences
        :param alignment: nested list of str fragmented previous alignment sequences
        :param sigma: int penalty for opening a gap
        :param epsilon: int penalty for extending a gap
        :param all_domains: dict pre-constructed dictionary from UniProt
        :return: tuple (unaligned_seqs: list of unaligned sequences excluding the newly added seq,
        alignment_new: list of new alignment sequences)
        """
        def update_alignment(alignment, align2_dummy, mode='string'):
            """
            :param mode: str "domain": input list, or "string": input string
            :param alignment: list of old alignment strings
            :param align2_dummy: str align2 generated recording information about new gaps to be added in old alignments
            :return: alignment_new: list of updated alignment strings with new gaps added
            """
            assert mode in ['domain', 'string'], 'Invalid Mode! mode in update alignment must be domain or string!'
            # alignment_new = []
            # base case: if input is a single string, not split by domains
            if mode == 'string':
                # init result as 1d list
                alignment_new = []
                for align in alignment:
                    align_new = ''
                    align_pos = 0
                    for pos in range(len(align2_dummy)):
                        if align2_dummy[pos] == '-':
                            align_new += '-'
                        else:
                            align_new += align[align_pos]
                            align_pos += 1
                    alignment_new.append(align_new)
            # if input is list split by domains, we do string mode for each fragment
            elif mode == 'domain':
                # init result as 2d list
                alignment_new = [[] for _ in range(len(alignment))]
                for frag_pos in range(len(alignment[0])):
                    # get a column in alignment
                    curr_align_ = [align[frag_pos] for align in alignment]
                    curr_dummy = align2_dummy[frag_pos]
                    # for each fragment, we can use string mode to get a column
                    new_align = update_alignment(curr_align_, curr_dummy, mode='string')
                    # now we have 1 column, but we need to append by row
                    for i in range(len(new_align)):
                        alignment_new[i].append(new_align[i])
            return alignment_new
        # find the best seq to add to alignment
        best_align1 = None
        best_align2 = None
        best_score = -inf
        best_raw_seq = None
        best_raw_id = None
        for i in range(len(unaligned_seqs)):
            curr_seq = unaligned_seqs[i]
            curr_id = unaligned_ids[i]
            # get domain info of this seq
            seq_structure_dict = sequence_to_domain_structure(curr_id, curr_seq, all_domains)
            # split this seq by domain info, this should have same structure as previous alignment
            seq_split = split_seq_by_structure_dict(seq_structure_dict, curr_seq)
            ##########
            assert len(seq_split) == len(alignment[0])  # alignment is now 2d list so we use [0]
            ##########
            # init final score, we will add curr_score of each fragment in loop below
            final_seq_align = []
            final_aligment_align_dummy = []
            final_score = 0
            # iterate through domains & linkers, produce fragmented alignment in a list
            for i in range(len(seq_split)):
                # the first fragment of current sequence, a single string
                curr_seq_frag = seq_split[i]
                # get the first aligned fragment of all previous sequences, a list of strings
                curr_alignment_frag = [align[i] for align in alignment]
                # for seq2 required in affine_gap_penalties_pair_wise_alignment we use a dummy sequence
                seq_frag_align, alignment_frag_align_dummy, curr_score = affine_gap_penalties_pair_wise_alignment(curr_seq_frag, ''.join('X' for _ in range(len(curr_alignment_frag[0]))),
                                                                                                                  sigma, epsilon, mode='profile', alignment=curr_alignment_frag)
                # add results for this alignment for this fragment
                final_score += curr_score
                final_seq_align.append(seq_frag_align)
                final_aligment_align_dummy.append(alignment_frag_align_dummy)
            if final_score > best_score:
                best_align1 = final_seq_align
                best_align2 = final_aligment_align_dummy
                best_score = final_score
                best_raw_seq = curr_seq
                best_raw_id = curr_id
        # update the existing alignment based on new gap information recorded in align2_dummy,
        # we need dummy align2 to guide how to make gaps in previous alignments
        alignment_new = update_alignment(alignment, best_align2, mode='domain')
        # add this best align to the now gapped previous alignments
        alignment_new.append(best_align1)
        # update unaligned seqs and ids
        # contrary to updating 2d alignments, both are 1d lists, so we can simply remove
        unaligned_seqs.remove(best_raw_seq)
        unaligned_ids.remove(best_raw_id)
        ###################
        print(best_raw_id+' has been aligned! '+str(len(unaligned_ids))+' proteins to be aligned...')
        ###################
        print('-------------------------------------------------------')
        # we successfully expanded our alignment by 1!
        return unaligned_seqs, alignment_new
    ######################
    print('-------------------------------------------------------')
    print('Domain-based Greedy MSA started...')
    print('-------------------------------------------------------')
    print('Categorizing '+str(len(id_seq_dict))+' sequences')
    ######################
    # extract ids & seqs from input dictionary
    ids = list(id_seq_dict.keys())
    seqs = list(id_seq_dict.values())
    # split seqs into categories
    categories = categorize_seqs_by_domain_info(all_domains, ids, seqs)
    ######################
    print('Sequence Categorization Complete')
    print('-------------------------------------------------------')
    print(str(len(categories))+' distinct domain-combinations found, they are (separated by "---"): ')
    for key in list(categories.keys()):
        print(key+' : '+str(len(categories[key]['seqs']))+' sequences')
    print('-------------------------------------------------------')
    ######################
    # init alignment
    alignments = categories.copy()
    # greedy MSA for each categories
    for structure_identifier in categories:
        ######################
        print('Performing greedy MSA on '+str(len(categories[structure_identifier]['seqs']))+' sequences of structure '+structure_identifier)
        ######################
        # extract seqs & ids for this category
        curr_seqs = categories[structure_identifier]['seqs']
        curr_ids = categories[structure_identifier]['ids']
        ######################
        print('-------------------------------------------------------')
        print('Looking for best initial pair...')
        ######################
        best_pair, curr_align = init_best_pair_alignment_domain_based(curr_ids, curr_seqs,
                                                                      sigma, epsilon, all_domains)
        # check if this category contains only one sequence, if so, we record itself, proceed with next category
        if best_pair == -1:
            alignments[structure_identifier]['category_alignment'] = curr_seqs
            continue
        # we join fragments together to check for unaligned seqs & ids
        joined_best_pair = [''.join(align_arr) for align_arr in best_pair]
        unaligned_seqs = [seq_ for seq_ in curr_seqs if seq_ not in joined_best_pair]
        unaligned_ids = [id_ for id_ in curr_ids if id_seq_dict[id_] not in joined_best_pair]
        # while not all aligned
        ######################
        if unaligned_seqs:
            print('Extending alignment '+structure_identifier+'...')
            print('-------------------------------------------------------')
        ######################
        while unaligned_seqs:
            unaligned_seqs, curr_align = add_one_seq_to_alignment_domain_based(unaligned_ids,
                                                                               unaligned_seqs, curr_align,
                                                                               sigma, epsilon, all_domains)
        # now the alignment process finished, our curr_align is still fragmented (2d list instead of 1d)
        # IF WE WANT DIRECT OUTPUT, we concatenate fragmented alignment right now, but this cost domain information
        # I'm not sure on 2022.11.27, so I concatenated anyway, different modes could be set later
        concat_final_alignment = [''.join(align_list) for align_list in curr_align]
        # record alignment for this category
        alignments[structure_identifier]['category_alignment'] = concat_final_alignment
    ######################
    print('-------------------------------------------------------')
    print('Domain-based Greedy MSA finished!')
    print('-------------------------------------------------------')
    ######################
    return alignments


"""
TEST CASES
"""


# MAIN TEST: Hugo's CRK
if __name__ == '__main__':
    # parsing data & uniprot domain info
    crk_data = parse_txt_to_dict('ICA/BMI3_domain_aware_aligner/CRK_aln.txt')
    all_domains_crk = parse_panda_to_dict('ICA/BMI3_domain_aware_aligner/uniprot_crk.tsv')
    # main algorithm
    domain_alignment_result = domain_aware_greedy_MSA(all_domains_crk, crk_data)
    # extracting alignments from result
    print('Outputing results: ')
    for structure in domain_alignment_result:
        print('-------------------------------------------------------')
        print('Alignment result for '+str(len(domain_alignment_result[structure]['seqs']))+' sequences of '+structure+' structure: ')
        for alignment in domain_alignment_result[structure]['category_alignment']:
            print(alignment)
    print('-------------------------------------------------------')

# # 2 seqs
# # two_seq_test_data_download_link = 'https://rosalind.info/problems/ba5j/'  <--------  DOWNLOAD LINK, YOU MAY NEED TO REGISTER & LOGIN
# def read_data(name):
#     with open('C:/Users/lenovo/Downloads/rosalind_'+name+'.txt','r') as infile:
#         return infile.readlines()
#
#
# data = read_data('ba5j (2)')
# seq1 = data[0].strip()
# seq2 = data[1].strip()
# for seq in affine_gap_penalties_pair_wise_alignment(seq1, seq2):
#     print(seq)
#
# # 3 seqs
# data = read_data('ba5m')
# seq1 = data[0].strip()
# seq2 = data[1].strip()
# seq3 = data[2].strip()
# alignment = greedy_multiple_alignment_with_affine_gap_penalties([seq1, seq2, seq3])
# for align in alignment:
#     print(align)
