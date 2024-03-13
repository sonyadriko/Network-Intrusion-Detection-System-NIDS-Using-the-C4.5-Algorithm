# Import library
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

# Load data (gantilah 'your_data.csv' dengan nama file dataset Anda)
# Pastikan format file dataset sesuai dengan kebutuhan
# Misalnya, menggunakan pandas:
import pandas as pd
# data = pd.read_csv('data25.xlsx')
data = pd.read_excel('data25.xlsx')

X = data.drop(['no.', 'service', 'attack_cat'], axis=1)
y = data['attack_cat']

# Pemisahan data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Membuat dan melatih model decision tree
dt_classifier = DecisionTreeClassifier(criterion='entropy', max_depth=None, random_state=42)
dt_classifier.fit(X_train, y_train)

# Melakukan prediksi pada data pengujian
y_pred = dt_classifier.predict(X_test)

# Evaluasi model
accuracy = metrics.accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Visualisasi decision tree (jika diinginkan)
from sklearn.tree import export_text, export_graphviz
import graphviz

feature_names = X.columns.tolist()

# Mendapatkan teks representasi decision tree
tree_rules = export_text(dt_classifier, feature_names=feature_names)

# Menampilkan hasil
print(tree_rules)

dot_data = export_graphviz(dt_classifier, out_file=None, 
                           feature_names=list(X.columns),
                           class_names=list(map(str, dt_classifier.classes_)),
                           filled=True, rounded=True, special_characters=True)  

graph = graphviz.Source(dot_data)
graph.render("decision_tree_visualization")
