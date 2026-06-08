import urllib.request
from predictor import get_predictor

genes = {'GAPDH': 'NM_002046.7', 'ACTB': 'NM_001101.5', 'TP53': 'NM_000546.6'}
predictor = get_predictor()

for name, acc in genes.items():
    url = f'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nucleotide&id={acc}&rettype=fasta&retmode=text'
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req) as response:
            fasta = response.read().decode('utf-8')
            seq = ''.join(fasta.split('\n')[1:])
            
            res = predictor.predict(seq)
            print(f"Gene: {name} | Prediction: {res['prediction']} | Confidence: {res['confidence']}% | AMR Prob: {res['probabilities']['amr']:.4f}")
    except Exception as e:
        print(f"Error fetching {name}: {e}")
