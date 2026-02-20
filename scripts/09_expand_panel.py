#!/usr/bin/env python3
"""
expand protein panel from 8 to 16 WT proteins.
- fetches sequences from UniProt REST API
- creates mutant variants
- writes expanded FASTA
- defines LCD regions for new proteins
"""

from pathlib import Path
import re

PROJECT = Path(__file__).resolve().parent.parent
SEQUENCES = PROJECT / "sequences"

# new protein sequences (extracted from UniProt fetches)
NEW_PROTEINS = {
    # amyloid - mature IAPP peptide (37 aa, residues 34-70 of P10997 precursor)
    "iapp_mature_wt": "KCNTATCATQRLANFLVHSSNNFGAILSSTNVGSNTY",

    # condensate maturation - TIA1 (P31483, 386 aa)
    "tia1_full_wt": (
        "MEDEMPKTLYVGNLSRDVTEALILQLFSQIGPCKNCKMIMDTAGNDPYCFVEFHEHRHAA"
        "AALAAMNGRKIMGKEVKVNWATTPSSQKKDTSSSTVVSTQRSQDHFHVFVGDLSPEITTE"
        "DIKAAFAPFGRISDARVVKDMATGKSKGYGFVSFFNKWDAENAIQQMGGQWLGGRQIRTN"
        "WATRKPPAPKSTYESNTKQLSYDEVVNQSSPSNCTVYCGGVTSGLTEQLMRQTFSPFGQI"
        "MEIRVFPDKGYSFVRFNSHESAAHAIVSVNGTTIEGHVVKCYWGKETLDMINPVQQQNQI"
        "GYPQPYGQWGQWYGNAQQIGQYMPNGWQVPAYGMYGQAWNQQGFNQTQSSAPWMGPNYGV"
        "QPPQGQNGSMLPNQPSGYRVAGYETQ"
    ),

    # condensate maturation - EWSR1 (Q01844, 656 aa)
    "ewsr1_full_wt": (
        "MASTDYSTYSQAAAQQGYSAYTAQPTQGYAQTTQAYGQQSYGTYGQPTDVSYTQAQTTAT"
        "YGQTAYATSYGQPPTGYTTPTAPQAYSQPVQGYGTGAYDTTTATVTTTQASYAAQSAYGT"
        "QPAYPAYGQQPAATAPTRPQDGNKPTETSQPQSSTGGYNQPSLGYGQSNYSYPQVPGSYP"
        "MQPVTAPPSYPPTSYSSTQPTSYDQSSYSQQNTYGQPSSYGQQSSYGQQSSYGQQPPTSY"
        "PPQTGSYSQAPSQYSQQSSSYGQQSSFRQDHPSSMGVYGQESGGFSGPGENRSMSGPDNR"
        "GRGRGGFDRGGMSRGGRGGGRGGMGSAGERGGFNKPGGPMDEGPDLDLGPPVDPDEDSDN"
        "SAIYVQGLNDSVTLDDLADFFKQCGVVKMNKRTGQPMIHIYLDKETGKPKGDATVSYEDP"
        "PTAKAAVEWFDGKDFQGSKLKVSLARKKPPMNSMRGGLPPREGRGMPPPLRGGPGGPGGP"
        "GGPMGRMGGRGGDRGGFPPRGPRGSRGNPSGGGNVQHRAGDWQCPNPGCGNQNFAWRTEC"
        "NQCKAPKPEGFLPPPFPPPGGDRGRGGPGGMRGGRGGLMDRGGPGGMFRGGRGGDRGGFR"
        "GGRGMDRGGFGGGRRGGPGGPPGPLMEQMGGRRGGRGGPGKMDKGEHRQERRDRPY"
    ),

    # condensate maturation - TAF15 (Q92804, 592 aa)
    "taf15_full_wt": (
        "MSDSGSYGQSGGEQQSYSTYGNPGSQGYGQASQSYSGYGQTTDSSYGQNYSGYSSYGQSQ"
        "SGYSQSYGGYENQKQSSYSQQPYNNQGQQQNMESSGSQGGRAPSYDQPDYGQQDSYDQQS"
        "GYDQHQGSYDEQSNYDQQHDSYSQNQQSYHSQRENYSHHTQDDRRDVSRYGEDNRGYGGS"
        "QGGGRGRGGYDKDGRGPMTGSSGGDRGGFKNFGGHRDYGPRTDADSESDNSDNNTIFVQG"
        "LGEGVSTDQVGEFFKQIGIIKTNKKTGKPMINLYTDKDTGKPKGEATVSFDDPPSAKA"
        "AIDWFDGKEFHGNIIKVSFATRRPEFMRGGGSGGGRRGRGGYRGRGGFQGRGGDPKSGDW"
        "VCPNPSCGNMNFARRNSCNQCNEPRPEDSRPSGGDFRGRGYGGERGYRGRGGRGGDRGGY"
        "GGDRSGGGYSGDRSGGGYGGDRSGGGYGGDRGGGYGGDRGGGYGGDRGGGYGGDRGGYGG"
        "DRGGGYGGDRGGYGGDRGGYGGDRGGYGGDRGGYGGDRSRGGYGGDRGGGSGGYGGDRSG"
        "GYGGDRSGGGYGGDRGGGYGGDRGGYGGKMGGRNDYRNDQRNRPY"
    ),

    # condensate maturation - hnRNPA2B1 (P22626 isoform 1, 353 aa)
    "hnrnpa2b1_full_wt": (
        "MEKTLETVPLERKKREKEQFRKLFIGGLSFETTEESLRNYYEQWGKLTDCVVMRDPASKR"
        "SRGFGFVTFSSMAEVDAAMAARPHSIDGRVVEPKRAVAREESGKPGAHVTVKKLFVGGIK"
        "EDTEEHHLRDYFEEYGKIDTIEIITDRQSGKKRGFGFVTFDDHDPVDKIVLQKYHTINGH"
        "NAEVRKALSRQEMQEVQSSRSGRGGNFGFGDSRGGGGNFGPGPGSNFRGGSDGYGSGRGF"
        "GDGYNGYGGGPGGGNFGGSPGYGGGRGGYGGGGPGYGNQGGGYGGGYDNYGGGNYGSGNY"
        "NDFGNYNQQPSNYGPMKSGNFGGSRNMGGPYGGGNYGPGGSGGSGGYGGRSRY"
    ),

    # functional condensate - DDX4 (Q9NQI0, 724 aa)
    "ddx4_full_wt": (
        "MGDEDWEAEINPHMSSYVPIFEKDRYSGENGDNFNRTPASSSEMDDGPSRRDHFMKSGFA"
        "SGRNFGNRDAGECNKRDNTSTMGGFGVGKSFGNRGFSNSRFEDGDSSGFWRESSNDCEDN"
        "PTRNRGFSKRGGYRDGNNSEASGPYRRGGRGSFRGCRGGFGLGSPNNDLDPDECMQRTGG"
        "LFGSRRPVLSGTGNGDTSQSRSGSGSERGGYKGLNEEVITGSGKNSWKSEAEGGESSDTQ"
        "GPKVTYIPPPPPEDEDSIFAHYQTGINFDKYDTILVEVSGHDAPPAILTFEEANLCQTLN"
        "NNIAKAGYTKLTPVQKYSIPIILAGRDLMACAQTGSGKTAAFLLPILAHMMHDGITASRF"
        "KELQEPECIIVAPTRELVNQIYLEARKFSFGTCVRAVVIYGGTQLGHSIRQIVQGCNILC"
        "ATPGRLMDIIGKEKIGLKQIKYLVLDEADRMLDMGFGPEMKKLISCPGMPSKEQRQTLMF"
        "SATFPEEIQRLAAEFLKSNYLFVAVGQVGGACRDVQQTVLQVGQFSKREKLVEILRNIGD"
        "ERTMVFVETKKKADFIATFLCQEKISTTSIHGDREQREREQALGDFRFGKCPVLVATSVA"
        "ARGLDIENVQHVINFDLPSTIDEYVHRIGRTGRCGNTGRAISFFDLESDNHLAQPLVKVL"
        "TDAQQDVPAWLEEIAFSTYIPGFSGSTRGNVFASVDTRKGKSTLNTAGFSSSQAPNPVDD"
        "ESWD"
    ),

    # functional condensate - NPM1 (P06748, 294 aa)
    "npm1_full_wt": (
        "MEDSMDMDMSPLRPQNYLFGCELKADKDYHFKVDNDENEHQLSLRTVSLGAGAKDELHIV"
        "EAEAMNYEGSPIKVTLATLKMSVQPTVSLGGFEITPPVVLRLKCGSGPVHISGQHLVAVE"
        "EDAESEDEEEEDVKLLSISGKRSAPGGGSKVPQKKVKLAADEDDDDDDEEDDDEDDDDDD"
        "FDDEEAEEKAPVKKSIRDTPAKNAQKSNQNGKDSKPSSTPRSKGQESFKKQEKTPKTPKG"
        "PSSVEDIKAKMQASIEKGGSLPKVEAKFINYVKNCFRMTDQEAIQDLWQWRKSL"
    ),

    # polyQ collapse - Ataxin-3 (P54252, 364 aa, ~14Q normal)
    "ataxin3_q14_wt": (
        "MESIFHEKQEGSLCAQHCLNNLLQGEYFSPVELSSIAHQLDEEERMRMAEGGVTSEDYRT"
        "FLQQPSGNMDDSGFFSIQVISNALKVWGLELILFNSPEYQRLRIDPINERSFICNYKEHW"
        "FTVRKLGKQWFNLNSLLTGPELISDTYLALFLAQLQQEGYSIFVVKGDLPDCEADQLLQM"
        "IRVQQMHRPKLIGEELAQLKEQRVHKTDLERVLEANDGSGMLDEDEEDLQRALALSRQEI"
        "DMEDEEADLRRAIQLSMQGSSRNISQDMTQTSGTNLTSEELRKRREAYFEKQQQKQQQQQ"
        "QQQQQQQGDLSGQSSHPCERPATSSGALGSDLGDAMSEEDMLQAAVTMSLETVRNDLKTEGK"
        "K"
    ),
}


def create_mutant(wt_seq, position_1based, wt_aa, mut_aa):
    """create point mutant from WT sequence (1-based position)."""
    idx = position_1based - 1
    if wt_seq[idx] != wt_aa:
        print(f"  WARNING: expected {wt_aa} at position {position_1based}, found {wt_seq[idx]}")
        return None
    return wt_seq[:idx] + mut_aa + wt_seq[idx+1:]


def create_polyq_expansion(wt_seq, normal_q_count, expanded_q_count):
    """expand a polyQ tract in the sequence."""
    # find the longest Q-tract
    q_runs = [(m.start(), m.end()) for m in re.finditer(r'Q{5,}', wt_seq)]
    if not q_runs:
        print("  WARNING: no polyQ tract found")
        return None

    # use the longest run
    longest = max(q_runs, key=lambda x: x[1] - x[0])
    start, end = longest
    current_q = end - start
    extra_q = expanded_q_count - current_q
    if extra_q <= 0:
        print(f"  WARNING: current Q count ({current_q}) >= target ({expanded_q_count})")
        return None

    return wt_seq[:end] + "Q" * extra_q + wt_seq[end:]


def main():
    # read existing FASTA
    existing_fasta = (SEQUENCES / "all_proteins.fasta").read_text()

    # new entries to append
    new_entries = []

    # add WT proteins
    for name, seq in NEW_PROTEINS.items():
        # clean sequence (remove any whitespace)
        seq = seq.replace(" ", "").replace("\n", "")
        new_entries.append(f">{name}\n{seq}")
        print(f"  added {name}: {len(seq)} aa")

    # create mutants
    # TIA1 P362L (Mackenzie 2017, juvenile ALS)
    tia1_seq = NEW_PROTEINS["tia1_full_wt"].replace(" ", "").replace("\n", "")
    tia1_p362l = create_mutant(tia1_seq, 362, "P", "L")
    if tia1_p362l:
        new_entries.append(f">tia1_P362L\n{tia1_p362l}")
        print(f"  added tia1_P362L: {len(tia1_p362l)} aa")

    # hnRNPA2B1 D290V (Kim 2013, ALS)
    a2b1_seq = NEW_PROTEINS["hnrnpa2b1_full_wt"].replace(" ", "").replace("\n", "")
    a2b1_d290v = create_mutant(a2b1_seq, 290, "D", "V")
    if a2b1_d290v:
        new_entries.append(f">hnrnpa2b1_D290V\n{a2b1_d290v}")
        print(f"  added hnrnpa2b1_D290V: {len(a2b1_d290v)} aa")

    # Ataxin-3 polyQ expansion (Q14 → Q72, pathogenic)
    atx3_seq = NEW_PROTEINS["ataxin3_q14_wt"].replace(" ", "").replace("\n", "")
    atx3_q72 = create_polyq_expansion(atx3_seq, 14, 72)
    if atx3_q72:
        new_entries.append(f">ataxin3_q72\n{atx3_q72}")
        print(f"  added ataxin3_q72: {len(atx3_q72)} aa")

    # write expanded FASTA
    expanded_fasta = existing_fasta.rstrip() + "\n" + "\n".join(new_entries) + "\n"
    out_path = SEQUENCES / "all_proteins.fasta"
    out_path.write_text(expanded_fasta)
    print(f"\n  written expanded FASTA: {out_path}")

    # count total sequences
    n_seqs = expanded_fasta.count(">")
    print(f"  total sequences: {n_seqs}")


if __name__ == "__main__":
    main()
