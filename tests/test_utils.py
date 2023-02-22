from progen.utils import download_checkpoint, fasta2dict

def test_load_fasta_file():
    fasta_file = "./data/uniprot_sprot.fasta"
    protein_dict = fasta2dict(fasta_file)

    assert len(protein_dict) > 0
    assert "id" in protein_dict[0]
    assert "desc" in protein_dict[0]
    assert "seq" in protein_dict[0]