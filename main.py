import sys
import pandas as pd
import knn
import naive_bayes

df_train = pd.read_csv(sys.argv[1])
df_validation = pd.read_csv(sys.argv[2])

if sys.argv[0] == "knn.py":
    # Neighbors: 5
    knn.knn(df_train, df_validation, 5)
elif sys.argv[0] == "naive_bayes.py":
    naive_bayes.naive_bayes(df_train, df_validation)
else:
    print("Perintah tidak valid. Silakan coba lagi.")