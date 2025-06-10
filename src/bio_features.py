import numpy as np

def gc_content(seq: str) -> float:
    """Calculate GC content percentage of an RNA sequence."""
    seq = seq.upper()
    gc = seq.count('G') + seq.count('C')
    valid = sum(seq.count(n) for n in 'AUCG')
    return gc / valid if valid > 0 else 0.0

def at_content(seq: str) -> float:
    """Calculate AT (AU) content percentage of an RNA sequence."""
    seq = seq.upper()
    at = seq.count('A') + seq.count('U')
    valid = sum(seq.count(n) for n in 'AUCG')
    return at / valid if valid > 0 else 0.0

def melting_temp(seq: str) -> float:
    """Estimate melting temperature (Tm) for short RNA sequences (Wallace rule)."""
    seq = seq.upper()
    a = seq.count('A')
    u = seq.count('U')
    g = seq.count('G')
    c = seq.count('C')
    return 2 * (a + u) + 4 * (g + c)

def length(seq: str) -> int:
    """Return the length of the sequence (excluding non-AUCG bases)."""
    seq = seq.upper()
    return sum(seq.count(n) for n in 'AUCG')

def base_frequencies(seq: str) -> np.ndarray:
    """Return the frequency of each base (A, U, C, G) as a numpy array."""
    seq = seq.upper()
    total = sum(seq.count(n) for n in 'AUCG')
    if total == 0:
        return np.zeros(4)
    return np.array([seq.count('A'), seq.count('U'), seq.count('C'), seq.count('G')]) / total

def purine_content(seq: str) -> float:
    """Calculate purine (A+G) content percentage of an RNA sequence."""
    seq = seq.upper()
    pur = seq.count('A') + seq.count('G')
    valid = sum(seq.count(n) for n in 'AUCG')
    return pur / valid if valid > 0 else 0.0

def pyrimidine_content(seq: str) -> float:
    """Calculate pyrimidine (C+U) content percentage of an RNA sequence."""
    seq = seq.upper()
    pyr = seq.count('C') + seq.count('U')
    valid = sum(seq.count(n) for n in 'AUCG')
    return pyr / valid if valid > 0 else 0.0

def molecular_weight(seq: str) -> float:
    """Estimate the molecular weight of an RNA sequence (in Daltons)."""
    # Average weights: A=329.2, U=306.2, C=305.2, G=345.2 (approx, without 5' and 3' phosphates)
    seq = seq.upper()
    weights = {'A': 329.2, 'U': 306.2, 'C': 305.2, 'G': 345.2}
    return sum(seq.count(base) * weight for base, weight in weights.items())

def dinucleotide_frequencies(seq: str) -> np.ndarray:
    """Return the frequency of each dinucleotide (AA, AU, ..., GG) as a numpy array (16 elements)."""
    seq = seq.upper()
    dinucs = [a + b for a in 'AUCG' for b in 'AUCG']
    total = len(seq) - 1
    if total <= 0:
        return np.zeros(16)
    counts = [sum(1 for i in range(total) if seq[i:i+2] == d) for d in dinucs]
    return np.array(counts) / total

def shannon_entropy(seq: str) -> float:
    """Calculate the Shannon entropy of the sequence (complexity measure)."""
    seq = seq.upper()
    from math import log2
    total = sum(seq.count(n) for n in 'AUCG')
    if total == 0:
        return 0.0
    probs = [seq.count(n) / total for n in 'AUCG']
    return -sum(p * log2(p) for p in probs if p > 0)

def longest_mononucleotide_run(seq: str) -> int:
    """Return the length of the longest run of a single nucleotide in the sequence."""
    seq = seq.upper()
    max_run = 0
    for base in 'AUCG':
        runs = [len(run) for run in seq.split(base) if run == '']
        if runs:
            max_run = max(max_run, max(runs))
    return max_run

def au_gc_ratio(seq: str) -> float:
    """Return the AU/GC ratio of the sequence."""
    seq = seq.upper()
    au = seq.count('A') + seq.count('U')
    gc = seq.count('G') + seq.count('C')
    return (au / gc) if gc > 0 else 0.0

def gc_skew(seq: str) -> float:
    """Calculate GC skew: (G - C) / (G + C)."""
    seq = seq.upper()
    g = seq.count('G')
    c = seq.count('C')
    return (g - c) / (g + c) if (g + c) > 0 else 0.0

def at_skew(seq: str) -> float:
    """Calculate AT skew: (A - U) / (A + U)."""
    seq = seq.upper()
    a = seq.count('A')
    u = seq.count('U')
    return (a - u) / (a + u) if (a + u) > 0 else 0.0

def unique_kmers(seq: str, k: int = 2) -> int:
    """Return the number of unique k-mers in the sequence."""
    seq = seq.upper()
    if len(seq) < k:
        return 0
    kmers = set(seq[i:i+k] for i in range(len(seq)-k+1))
    return len(kmers)

def reverse_complement(seq: str) -> str:
    """Return the reverse complement of the RNA sequence."""
    seq = seq.upper()
    complement = str.maketrans('AUCG', 'UAGC')
    return seq.translate(complement)[::-1]

def is_palindromic(seq: str) -> bool:
    """Check if the sequence is palindromic (equal to its reverse complement)."""
    seq = seq.upper()
    return seq == reverse_complement(seq)

def count_ambiguous(seq: str) -> int:
    """Count the number of ambiguous (non-AUCG) bases in the sequence."""
    seq = seq.upper()
    return sum(1 for base in seq if base not in 'AUCG')
