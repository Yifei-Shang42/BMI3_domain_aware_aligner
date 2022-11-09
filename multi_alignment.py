"""
BMI3 Group5 Project4: Domain-aware aligner
"""

import numpy as np
from math import inf
import blosum as bl
mat = bl.BLOSUM(62)

"""
PAIR-WISE GLOBAL ALIGNMENT WITH AFFINE GAP PENALTY
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
        :param profile:
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
GREEDY ALGO FOR MSA BASED ON PAIR-WISE GLOBAL ALIGNMENT
"""


def greedy_multiple_alignment_with_affine_gap_penalties(seqs, sigma=11, epsilon=1):
    def init_best_pair_alignment(seqs, sigma=11, epsilon=1):
        """
        :param seqs: list collection of multiple input protein sequences
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
TEST CASES
"""


# 2 seqs
# two_seq_test_data_download_link = 'https://rosalind.info/problems/ba5j/'  <--------  DOWNLOAD LINK, YOU MAY NEED TO REGISTER & LOGIN
def read_data(name):
    with open('C:/Users/lenovo/Downloads/rosalind_'+name+'.txt','r') as infile:
        return infile.readlines()


data = read_data('ba5j (2)')
seq1 = data[0].strip()
seq2 = data[1].strip()
for seq in affine_gap_penalties_pair_wise_alignment(seq1, seq2):
    print(seq)

# 3 seqs
data = read_data('ba5m')
seq1 = data[0].strip()
seq2 = data[1].strip()
seq3 = data[2].strip()
alignment = greedy_multiple_alignment_with_affine_gap_penalties([seq1, seq2, seq3])
for align in alignment:
    print(align)


