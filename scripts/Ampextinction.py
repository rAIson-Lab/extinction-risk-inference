import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../CONFOLD/'))) #add CONFOLD to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) #add the parent directory to the path

import random
random.seed(42)
import numpy as np
from foldrm import Classifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd
from datasets import extinction_amphib # Our new function



model, data = extinction_amphib()

from utils import split_data

train_data, test_data = split_data(data, ratio=0.9, shuffle=True)

# --- Balanceo de clases por submuestreo ---
from collections import defaultdict
def balancear_clases(data):
     clases = defaultdict(list)
     for row in data:
          clases[row[-1]].append(row)
     min_count = min(len(v) for v in clases.values())
     balanced = []
     for v in clases.values():
          balanced.extend(random.sample(v, min_count))
     random.shuffle(balanced)
     return balanced

train_data = balancear_clases(train_data)

# Training

model.fit(train_data, ratio=0.9)
model.confidence_fit(train_data, improvement_threshold=0.9)

print("\nLearned Answer Set Program rules:\n")

model.print_asp()

# --- Ranking de variables más usadas en reglas ---
from collections import Counter
import re

def contar_variables_en_reglas(asp_rules, attrs):
     var_counter = Counter()
     # Buscar nombres de atributos en cada regla ASP
     for rule in asp_rules:
          for attr in attrs:
               # Buscar el nombre del atributo en formato bajo (como en las reglas)
               attr_key = attr.lower().replace(' ', '_')
               # Buscar coincidencias tipo attr_key(X,...) en la regla
               if re.search(rf'\b{attr_key}\s*\(', rule):
                    var_counter[attr] += 1
     return var_counter

asp_rules = model.asp()
var_counter = contar_variables_en_reglas(asp_rules, model.attrs)
print("\nRanking de variables más usadas en las reglas aprendidas:")

# Mostrar todas las variables, incluso las que no aparecen en ninguna regla
for var in model.attrs:
     count = var_counter.get(var, 0)
     print(f"{var}: {count} apariciones")


Y_pred = model.predict(test_data)

print("\nEjemplo de predicciones (primeros 10):")
for i, (pred, obs) in enumerate(zip(Y_pred[:10], test_data[:10])):
     print(f"Obs {i+1}: pred = {pred}, entrada = {obs}")

# Matriz de confusión

pred_classes = [p[0] for p in Y_pred if p is not None and p[0] is not None]
true_classes = [row[-1] for p, row in zip(Y_pred, test_data) if p is not None and p[0] is not None]

all_labels = sorted(list(set(true_classes + pred_classes)))

if pred_classes:
     cm = confusion_matrix(true_classes, pred_classes, labels=all_labels)
     df_cm = pd.DataFrame(cm, index=all_labels, columns=all_labels)
     print("\nMatriz de confusión:")
     print(df_cm)
else:
     print("\nNo hay predicciones válidas para matriz de confusión.")