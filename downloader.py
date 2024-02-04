import gdown
import tarfile
import os

url = "https://drive.google.com/uc?id=1BM5Oevom-zo1bhrF9i_qfDHiO0Tzv8BO"

fn = "Data.tar.gz"
gdown.download(url, fn)

file = tarfile.open(fn)
file.extractall(".")
os.remove("Data.tar.gz")