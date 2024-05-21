import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np

# 1. Wczytanie danych
# Załaduj dane z pliku CSV (przykład)
df = pd.read_csv('data.csv')

# Załóżmy, że ostatnia kolumna to etykieta decyzyjna
X = df.iloc[:, :-1]  # cechy (atrybuty)
y = df.iloc[:, -1]   # etykiety (klasy decyzyjne)

# 2. Obsługa brakujących danych
# Użyj SimpleImputer do wypełnienia brakujących wartości
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# Definiowanie transformacji dla cech numerycznych i kategorycznych
numeric_transformer = SimpleImputer(strategy='mean')
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Użycie ColumnTransformer do przetworzenia kolumn
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Tworzenie potoku przetwarzania i klasyfikatora
clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(random_state=42))
])

# 3. Podział danych na zestaw treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 4. Trenowanie modelu
clf.fit(X_train, y_train)

# Pobieranie ważności cech z modelu drzewa decyzyjnego
# Dopasowanie preprocesora i pobieranie przetworzonych danych
clf.named_steps['preprocessor'].fit(X_train)
X_train_processed = clf.named_steps['preprocessor'].transform(X_train)

# Uzyskiwanie nazw cech
# Najpierw dopasuj OneHotEncoder osobno, aby uzyskać nazwy cech
onehot = clf.named_steps['preprocessor'].named_transformers_['cat']['onehot']
onehot.fit(X[categorical_features])
categorical_feature_names = onehot.get_feature_names_out(categorical_features).tolist()

# Łączenie nazw cech numerycznych i kategorycznych
feature_names = numeric_features.tolist() + categorical_feature_names
importances = clf.named_steps['classifier'].feature_importances_

# Utwórz DataFrame z ważnościami cech
feature_ranking = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
})

# Posortuj cechy według ważności malejąco
feature_ranking = feature_ranking.sort_values(by='Importance', ascending=False)

print("Ranking atrybutów:")
print(feature_ranking)

# Opcjonalnie: wybór najważniejszych cech
selector = SelectFromModel(clf.named_steps['classifier'], prefit=True)
X_important_train = selector.transform(X_train_processed)
X_important_test = selector.transform(clf.named_steps['preprocessor'].transform(X_test))

# Sprawdź wybrane najważniejsze cechy
important_features = [feature_names[i] for i in np.where(selector.get_support())[0]]
print("\nNajważniejsze cechy:")
print(important_features)

# Można teraz trenować model na wybranych cechach
clf_important = DecisionTreeClassifier(random_state=42)
clf_important.fit(X_important_train, y_train)

# Ewaluacja modelu
score = clf_important.score(X_important_test, y_test)
print("\nDokładność modelu na wybranych cechach:", score)

# Wypisywanie reguł decyzyjnych
# Przetworzenie denych testowych w celu uzyskania płnyh nazw cech
X_test_processed = clf.named_steps['preprocessor'].transform(X_test)

# Tworzenie i trenowanie pełnego drzewa decyzyjnego na pełnych danych
full_tree_clf = DecisionTreeClassifier(random_state=42)
full_tree_clf.fit(X_test_processed, y_test)

# Wypisywanie reguł decyzyjnych
tree_rules = export_text(full_tree_clf, feature_names=feature_names)
print("\nReguły decyzyjne drzewa decyzyjnego:")
print(tree_rules)