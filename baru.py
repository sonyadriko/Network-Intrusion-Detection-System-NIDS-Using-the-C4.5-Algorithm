from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz
from sklearn.model_selection import train_test_split
import pandas as pd

# Your dataset (replace this with your actual dataset)
df = pd.read_excel('data20.xlsx')

# Drop unnecessary columns (replace with the actual columns you need)
X = df.drop(['no.'], axis=1)

# Convert categorical variables to numerical using one-hot encoding
X = pd.get_dummies(X)

# Target variable
y = df['attack_cat']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a Decision Tree Classifier with different parameters
model = DecisionTreeClassifier(criterion='entropy', random_state=42, min_samples_split=144)
model.fit(X_train, y_train)

# Visualize the Decision Tree
dot_data = export_graphviz(model, out_file=None,
                           feature_names=X.columns,
                           class_names=y.unique(),
                           filled=True, rounded=True, special_characters=True)

graph = graphviz.Source(dot_data)
graph.render("decision_tree")
graph.view("decision_tree")
