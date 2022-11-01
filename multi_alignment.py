"""
BMI3 Group5 Project4: Domain-aware aligner
"""


import numpy as np
from math import inf

"""
PAIR-WISE MULTIPLE ALIGNMENT WITH AFFINE GAP PENALTY
"""

def affine_gap_penalties_pair_wise_alignment(seq1, seq2, sigma=11, epsilon=1, penalty=None):
    """
    :param seq1: str input protein sequence 1
    :param seq2: str input protein sequence 2
    :param sigma: int penalty for opening a gap
    :param epsilon: int penalty for extending a gap
    :return: tuple (align1: str alignment for seq1, align2: str alignment for seq2, score: float best alignment score)
    """
    def gen_backtrack(seq1, seq2, sigma=11, epsilon=1):
        """
        :param seq1: str input protein sequence 1
        :param seq2: str input protein sequence 2
        :param sigma: int penalty for opening a gap
        :param epsilon: int penalty for extending a gap
        :return: tuple (backtracks: list [3 np.array backtracks for gap1, match, and gap2], score: float best alignment score)
        """
        def read_penalty(path):
            """
            :param path: str file path for penalty matrix
            :return: list of lines in penalty matrix
            """
            with open(path, 'r') as infile:
                return infile.readlines()
        # init penalty
        penalty = read_penalty('C:/Users/lenovo/Downloads/rosalind_penalty.txt')
        penalty_matrix = [[eval(num) for num in line.strip().split()] for line in penalty]
        aa_idx = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9, 'M': 10,
                       'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19}
        len1 = len(seq1)
        len2 = len(seq2)
        # init scores
        match_scores = -inf * np.ones((len1+1, len2+1))
        gap1_scores = -inf * np.ones((len1+1, len2+1))
        gap2_scores = -inf * np.ones((len1+1, len2+1))
        # init backtracks
        match_back = np.zeros((len1+1, len2+1))
        gap1_back = np.zeros((len1+1, len2+1))
        gap2_back = np.zeros((len1+1, len2+1))
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
        # init lateral gap extensions
        for pos1 in range(2, len1 + 1):
            gap1_scores[pos1, 0] = gap1_scores[pos1 - 1, 0] - epsilon
            match_scores[pos1, 0] = gap1_scores[pos1, 0]
        for pos2 in range(2, len2 + 1):
            gap2_scores[0, pos2] = gap2_scores[0, pos2 - 1] - epsilon
            match_scores[0, pos2] = gap2_scores[0, pos2]
        # start dynamic traversing
        for pos1 in range(1, len1 + 1):
            for pos2 in range(1, len2 + 1):
                # to best get to gap 1, do we open/extend gap
                gap1_extend_score = gap1_scores[pos1 - 1, pos2] - epsilon
                gap1_open_score = match_scores[pos1 - 1, pos2] - sigma
                gap1_best_score = np.max([gap1_open_score, gap1_extend_score])
                gap1_scores[pos1, pos2] = gap1_best_score
                # if we should open gap:
                if gap1_best_score == gap1_open_score:
                    # we record at gap1 backtrack as from match
                    gap1_back[pos1][pos2] = 'match'
                else:
                    # if extend
                    gap1_back[pos1][pos2] = 'gap1'

                # to best get to gap 2, do we open/extend gap
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

                # do we continue with gaps / do a match
                match_match_score = match_scores[pos1 - 1, pos2 - 1] + \
                                    penalty_matrix[aa_idx[seq1[pos1-1]]][aa_idx[seq2[pos2-1]]]
                best_score = np.max([gap1_best_score, match_match_score, gap2_best_score])
                match_scores[pos1, pos2] = best_score
                if best_score == match_match_score:
                    match_back[pos1][pos2] = 'match'
                elif best_score == gap2_best_score:
                    match_back[pos1][pos2] = 'gap2'
                elif best_score == gap1_best_score:
                    match_back[pos1][pos2] = 'gap1'
        return [gap1_back, match_back, gap2_back], match_scores[len1, len2]
    def gen_alignment_from_backtrack(backtracks, seq1, seq2):
        """
        :param backtracks: list [3 np.array backtracks for gap1, match, and gap2]
        :param seq1: str input protein sequence 1
        :param seq2: str input protein sequence 2
        :return: tuple (align1: str alignment for seq1, align2: str alignment for seq2)
        """
        # start at the end of backtrack
        pos1 = len(seq1)
        pos2 = len(seq2)
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
    backtracks, score = gen_backtrack(seq1, seq2, sigma, epsilon)
    align1, align2 = gen_alignment_from_backtrack(backtracks, seq1, seq2)
    return align1[::-1], align2[::-1], score

#----------NOT IMPLEMENTED------------------#
def greedy_multiple_alignment_with_affine_gap_penalties(seqs, sigma=11, epsilon=1):
    def find_best_pair_alignment(seqs):
        best_pair = None
        best_score = -inf
        best_align = None
        for pos1 in range(len(seqs) - 1):
            for pos2 in range(pos1 + 1, len(seqs)):
                seq1 = seqs[pos1]
                seq2 = seqs[pos2]
                align1, align2, curr_score = affine_gap_penalties_pair_wise_alignment(seq1, seq2)
                if curr_score > best_score:
                    best_score = curr_score
                    best_pair = (seq1, seq2)
                    best_align = [align1, align2]
        return best_pair, best_align
    def gen_penalty_from_alignment(alignments):
        return
    # first, we find the pair with highest score
    best_pair, best_align = find_best_pair(seqs)
    # init all_alignment contains as the best pair alignment
    all_alignments = best_align
    # record aligned seqs
    aligned = list(best_pair)
    #--------------------------------------------#
    # STARTING FROM HERE, ALL ARE PSEUDO CODES - 2022.11.1 23:21
    #--------------------------------------------#
    # while not all aligned
    while len(aligned) < len(seqs):
        # generate penalty matrix based on current alignment
        curr_penalty = gen_penalty_from_alignment(all_alignments)
        # 'align' to profile by using specific penalty matrix
        affine_gap_penalties_pair_wise_alignment(seq1, 'A SEQUENCE OF SAME LENGTH OF all_alignment', curr_penalty)
    # alignment_pro
    # keep track which ones aligned
#----------NOT IMPLEMENTED------------------#


"""
TEST CASES
"""
# 2 seqs
# two_seq_test_data_download_link = 'https://rosalind.info/problems/ba5j/'  <--------  DOWNLOAD LINK, YOU MAY NEED TO REGISTER & LOGIN
data = read_data('ba5j (2)')
seq1 = data[0].strip()
seq2 = data[1].strip()
for seq in affine_gap_penalties_pair_wise_alignment(seq1, seq2):
    print(seq)

# 3 seqs
# data = read_data('ba5m')
# seq1 = data[0].strip()
# seq2 = data[1].strip()
# seq3 = data[2].strip()


