from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import graphviz
import pandas as pd

# Data yang diberikan
# data = {
#     'no.': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
#     'service': ['http', 'dns', 'dns', 'dns', 'dns', 'dns', 'dns', 'dns', 'dns', 'dns', 'http', 'ftp', 'http', 'ftp', 'http', 'http', 'dns', 'http', 'dns', 'dns'],
#     'spkts': [10.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 8.0, 12.0, 10.0, 14.0, 10.0, 42.0, 2.0, 10.0, 2.0, 2.0],
#     'sbytes': [776.0, 114.0, 114.0, 114.0, 114.0, 114.0, 114.0, 114.0, 114.0, 132.0, 1028.0, 572.0, 826.0, 756.0, 766.0, 47218.0, 114.0, 830.0, 114.0, 114.0],
#     'sttl': [62.0, 254.0, 254.0, 254.0, 254.0, 254.0, 254.0, 254.0, 254.0, 31.0, 31.0, 254.0, 62.0, 254.0, 62.0, 254.0, 254.0, 62.0, 254.0, 254.0],
#     'smean': [78.0, 57.0, 57.0, 57.0, 57.0, 57.0, 57.0, 57.0, 57.0, 66.0, 129.0, 48.0, 83.0, 54.0, 77.0, 1124.0, 57.0, 83.0, 57.0, 57.0],
#     'attack_cat': ['Exploits', 'Generic', 'Generic', 'Generic', 'Generic', 'Generic', 'Generic', 'Generic', 'Generic', 'Normal', 'Normal', 'Normal', 'Normal', 'Fuzzers', 'Exploits', 'Exploits', 'Generic', 'Exploits', 'Generic', 'Generic']
# }

df = pd.read_excel('data20.xlsx')

# df = pd.DataFrame(data)

# Mengubah data kategorikal menjadi numerik
df_encoded = pd.get_dummies(df[['service']], prefix=['service'])

# Menggabungkan data numerik dengan data asli
df = pd.concat([df, df_encoded], axis=1)

# Memisahkan fitur dan label
X = df.drop(['no.', 'attack_cat', 'service'], axis=1)
y = df['attack_cat']

# Menghitung Gain Ratio untuk setiap atribut
gain_ratios = []
for column in X.columns:
    model = DecisionTreeClassifier(criterion='entropy')
    model.fit(X[[column]], y)
    gain_ratio = model.tree_.impurity[0] - model.tree_.impurity[1]
    gain_ratios.append((column, gain_ratio))

# Memilih atribut dengan Gain Ratio tertinggi sebagai akar pohon
# best_attribute = max(gain_ratios, key=lambda x: x[1])[0]
best_attribute = 'sbytes'

# Membuat model DecisionTreeClassifier dengan atribut terbaik sebagai akar
model = DecisionTreeClassifier(criterion='entropy', random_state=42)
model.fit(X[[best_attribute]], y)

# Membagi data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Membuat model DecisionTreeClassifier dengan atribut terbaik sebagai akar
model = DecisionTreeClassifier(criterion='entropy', random_state=42, min_samples_split=144)  # Sesuaikan dengan nilai yang diinginkan
model.fit(X_train[[best_attribute]], y_train)

# Memprediksi data uji
y_pred = model.predict(X_test[[best_attribute]])

# Menampilkan classification report tanpa peringatan UndefinedMetricWarning
print("Classification Report:")
print(classification_report(y_test, y_pred, zero_division=1))
# Menampilkan confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Menampilkan confusion matrix menggunakan heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
# Menampilkan pohon keputusan
# from sklearn.tree import export_text
# tree_rules = export_text(model, feature_names=[best_attribute])
# print(tree_rules)

# Menyimpan pohon keputusan sebagai file DOT
dot_data = export_graphviz(model, out_file=None, 
                           feature_names=[best_attribute],  
                           class_names=y.unique(),
                           filled=True, rounded=True, special_characters=True)  

# Menggunakan graphviz untuk membuat file PNG dari DOT
graph = graphviz.Source(dot_data)
graph.render("decision_tree")

# Menampilkan pohon keputusan sebagai gambar PNG
graph.view("decision_tree")