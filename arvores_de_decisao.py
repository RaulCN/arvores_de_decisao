import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# Introdução para o usuário
print("Este programa treina um classificador de árvore de decisão usando dados fictícios relacionados à geografia brasileira.")
print("Os dados incluem informações sobre população, PIB, desenvolvimento humano e se a região é uma capital.")
print("Em seguida, o programa faz previsões sobre a região com base nos novos dados fornecidos.")

# Dados de treinamento fictícios relacionados à geografia brasileira
data = [
    {'População': 'Alta', 'PIB': 'Alto', 'Desenvolvimento Humano': 'Alto', 'Capital': 'Sim', 'Região': 'Sudeste'},
    {'População': 'Média', 'PIB': 'Médio', 'Desenvolvimento Humano': 'Médio', 'Capital': 'Não', 'Região': 'Nordeste'},
    {'População': 'Baixa', 'PIB': 'Baixo', 'Desenvolvimento Humano': 'Baixo', 'Capital': 'Não', 'Região': 'Norte'},
    {'População': 'Alta', 'PIB': 'Alto', 'Desenvolvimento Humano': 'Alto', 'Capital': 'Sim', 'Região': 'Centro-Oeste'},
    {'População': 'Média', 'PIB': 'Médio', 'Desenvolvimento Humano': 'Médio', 'Capital': 'Não', 'Região': 'Sul'}
]

# Convertendo os dados em um formato adequado para o treinamento
X = np.array([[entry['População'], entry['PIB'], entry['Desenvolvimento Humano'], entry['Capital']] for entry in data])
y = np.array([entry['Região'] for entry in data])

# Convertendo as strings em números
label_encoders = []
for i in range(X.shape[1]):
    encoder = LabelEncoder()
    X[:, i] = encoder.fit_transform(X[:, i])
    label_encoders.append(encoder)

# Criando e treinando o classificador de árvore de decisão
clf = DecisionTreeClassifier()
clf.fit(X, y)

# Imprimindo a árvore de decisão treinada
print("\nÁrvore de Decisão treinada:")
print(clf)

# Dados fictícios para fazer previsões
new_data = [
    {'População': 'Alta', 'PIB': 'Alto', 'Desenvolvimento Humano': 'Alto', 'Capital': 'Não'},  # Nordeste
    {'População': 'Baixa', 'PIB': 'Baixo', 'Desenvolvimento Humano': 'Médio', 'Capital': 'Sim'},  # Centro-Oeste
    {'População': 'Média', 'PIB': 'Alto', 'Desenvolvimento Humano': 'Alto', 'Capital': 'Não'}  # Sudeste
]

# Convertendo os novos dados em um formato adequado para fazer previsões
X_new = np.array([[entry['População'], entry['PIB'], entry['Desenvolvimento Humano'], entry['Capital']] for entry in new_data])

# Convertendo os rótulos de string para números
for i, encoder in enumerate(label_encoders):
    X_new[:, i] = encoder.transform(X_new[:, i])

# Fazendo previsões com o classificador treinado
predictions = clf.predict(X_new)

# Imprimindo as previsões
print("\nPrevisões para os novos dados:")
for i, prediction in enumerate(predictions):
    print(f"Novo dado {i+1}: Região prevista: {prediction}")
