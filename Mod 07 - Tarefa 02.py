#!/usr/bin/env python
# coding: utf-8

# # Módulo 07 - Tarefa 02
# 
# #### 1) Carregue a base e garanta que a base está como deveria.
# 
# Considere a base que você ajustou na lição passada. Carregue-a. Caso ainda haja alguma pendência, ajuste - lembre-se de que o scikitlearn não recebe variáveis em formato string, somente numéricas, e não aceita '*missings*'. 
# 
# - Separe 70% da base para treinamento e 30% para validação. Cada uma dessas partes terá dois objetos, um armazenando a variável resposta ```mau```e outro armazenando as variáveis explicativas (lembrando: sem variáveis string, já com as dummies).

# In[3]:


import pandas as pd
import requests
from sklearn.model_selection import train_test_split
# importar o arquivo base.
data_base = pd.read_csv('dados/demo01.csv')
data_base.head()


# In[8]:


# Criar um DataFrame

data_base = pd.DataFrame({
        'id_cliente': range(100),
        'genero': ['M', 'F'] * 50,
        'renda': range(1000, 3000, 20),
        'tipo_residencia': ['propria', 'alugada'] * 50,
        'mau': [False, True, False, False] * 25
    })
data_base


# In[9]:


# Verificar valores ausentes ('missings')
print("Valores ausentes por coluna antes do tratamento:")
print(data_base.isnull().sum())


# In[10]:


#Remover linhas com valores ausentes se existir.
data_base = data_base.dropna()
print("\nBase de dados após remover linhas com valores ausentes. Shape:", data_base.shape)


# In[11]:


# Scikit-learn só aceita variáveis numéricas. Converter as colunas de texto.
# Primeiro, separamos a variável resposta ('y') das explicativas ('X').
X = data_base.drop('mau', axis=1)
y = data_base['mau']

# A variável resposta 'mau' deve ser numérica (0 ou 1).
# Se ela estiver como True/False, podemos convertê-la para 1/0.
y = y.astype(int)

# Agora, convertemos as variáveis explicativas de texto em 'dummies'.
X_dummies = pd.get_dummies(X, drop_first=True)

print("\nPré-visualização das variáveis explicativas após a criação de dummies:")
print(X_dummies.head())


# In[15]:


##Separar 70% para base de treinamento e 30% de validação##
   
# Usamos a função train_test_split do scikit-learn.
# 'test_size=0.3' garante a proporção 70/30.
# 'random_state' é usado para que a divisão seja sempre a mesma ao rodar o script.
X_treino, X_valid, y_treino, y_valid = train_test_split(X_dummies, y,test_size=0.3,random_state=42)


# In[16]:


# 3) Verificar os objetos criados

print("\n--- Divisão Concluída ---")
print(f"Shape das variáveis explicativas de treino (X_treino): {X_treino.shape}")
print(f"Shape da variável resposta de treino (y_treino): {y_treino.shape}")
print(f"Shape das variáveis explicativas de validação (X_valid): {X_valid.shape}")
print(f"Shape da variável resposta de validação (y_valid): {y_valid.shape}")


# #### 2) Vamos para o modelo:
# 
# 1. Defina um objeto com a função da árvore de decisão vista em aula.
# 2. Treine o modelo com os dados que você separou para treinamento.
# 3. Visualize a árvore. Talvez você queira aumentar um pouco a figura.
# 4. Produza uma visualização da matriz de classificação (ou matriz de confusão) - coloque os rótulos como "aprovados" e "reprovados" - pois afinal de contas, são essas as decisões que se podem tomar com propostas de crédito.
# 5. Calcule a acurácia na base de treinamento

# In[18]:


# Importando as bibliotecas necessárias para o modelo e visualizações
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# As variáveis X_treino, X_valid, y_treino, y_valid já existem, optidos na questão 1.

# 1. 
# Usaado o random_state para garantir que o resultado seja sempre o mesmo
arvore_decisao = DecisionTreeClassifier(random_state=42)

# 2. Treinado o modelo com os dados separados para treinamento
arvore_decisao.fit(X_treino, y_treino)
print("Modelo de Árvore de Decisão treinado com sucesso!")

# 3. Visualizada a árvore.
# Vamos extrair os nomes das colunas para legendar a árvore
nomes_das_features = X_treino.columns.tolist()

# Definido os nomes das classes (lembrando que 0=bom pagador, 1=mau pagador)
# Na matriz de confusão, a tarefa pede 'Aprovado' e 'Reprovado'
nomes_das_classes = ['Aprovado', 'Reprovado'] # 0 = Aprovado, 1 = Reprovado

plt.figure(figsize=(25, 12))  # Aumentando o tamanho da figura para melhor visualização
plot_tree(arvore_decisao,
          feature_names=nomes_das_features,
          class_names=nomes_das_classes,
          filled=True,      # Colore os nós para indicar a classe majoritária
          rounded=True,     # Deixa os nós com cantos arredondados
          fontsize=10)      # Define o tamanho da fonte
plt.title("Visualização da Árvore de Decisão")
plt.show()

# 4. Produzida uma visualização da matriz de confusão
# Primeiro, o modelo precisa fazer previsões na base de treino
y_pred_treino = arvore_decisao.predict(X_treino)

# Agora, criamos a matriz de confusão
matriz_confusao = confusion_matrix(y_treino, y_pred_treino)

# Usamos o Seaborn para uma visualização mais bonita
plt.figure(figsize=(8, 7))
sns.heatmap(matriz_confusao,
            annot=True,               # Mostra os números dentro de cada célula
            fmt="d",                  # Formata os números como inteiros
            cmap='Blues',             # Define a paleta de cores
            xticklabels=nomes_das_classes,
            yticklabels=nomes_das_classes)

plt.title('Matriz de Confusão - Base de Treinamento')
plt.ylabel('Verdadeiro (Real)')
plt.xlabel('Previsto (Modelo)')
plt.show()


# 5. Calcule a acurácia na base de treinamento
acuracia_treino = accuracy_score(y_treino, y_pred_treino)

print(f"\nA acurácia do modelo na base de TREINAMENTO é: {acuracia_treino:.4f}")
print(f"Ou seja, o modelo acerta {acuracia_treino*100:.2f}% das previsões nos dados de treino.")


# #### 3) Vamos avaliar o modelo na base de testes
# 
# 1. Classifique a base de teste de acordo com a árvore que você treinou no item 2.
# 2. Produza a visualização da matriz de confusão para a base de teste.
# 3. Calcule a acurácia da base de teste. Compare com a acurácia da base de treinamento.
# 4. Treine uma nova árvore com número mínimo de observações por folha de 5 e máximo de profundidade de 10. Use o random_state = 123. Avalie a matriz de classificação. Observe a distribuição da predição - qual a proporção de proponentes foram classificados como 'maus'?
# 5. Como ficaria a acurácia se você classificasse todos os contratos como 'bons'?

# In[20]:


import numpy as np

# As variáveis arvore_decisao, X_treino, y_treino, X_valid, y_valid,
# acuracia_treino e nomes_das_classes já existem no ambiente.

print("--- 3) Avaliando o modelo na base de testes ---")

# 1. Classificada a base de teste de acordo com a árvore treinada no item 2.
y_pred_valid = arvore_decisao.predict(X_valid)
print("Base de teste classificada com o primeiro modelo.")

# 2. Produzida a visualização da matriz de confusão para a base de teste.
matriz_confusao_valid = confusion_matrix(y_valid, y_pred_valid)

plt.figure(figsize=(8, 7))
sns.heatmap(matriz_confusao_valid,
            annot=True,
            fmt="d",
            cmap='Greens', # Mudando a cor para diferenciar do gráfico de treino
            xticklabels=nomes_das_classes,
            yticklabels=nomes_das_classes)

plt.title('Matriz de Confusão - Base de Teste (Validação)')
plt.ylabel('Verdadeiro (Real)')
plt.xlabel('Previsto (Modelo)')
plt.show()

# 3. Calcula a acurácia da base de teste. Compara com a acurácia da base de treinamento.
acuracia_valid = accuracy_score(y_valid, y_pred_valid)

print("\n--- Comparação de Acurácia ---")
print(f"Acurácia na base de TREINAMENTO: {acuracia_treino*100:.2f}%")
print(f"Acurácia na base de TESTE: {acuracia_valid*100:.2f}%")

if acuracia_treino > acuracia_valid:
    print("\nA acurácia no teste foi menor que no treino. Isso é comum e pode indicar algum sobreajuste (overfitting).")
else:
    print("\nA acurácia no teste foi similar ou maior que no treino, o que é um bom sinal de generalização.")


# 4. Treina uma nova árvore com parâmetros definidos e avalia.
print("\n--- Treinando uma nova árvore com hiperparâmetros ajustados ---")
arvore_tunada = DecisionTreeClassifier(max_depth=10,
                                       min_samples_leaf=5,
                                       random_state=123)
arvore_tunada.fit(X_treino, y_treino)

# Avalia a matriz de classificação da nova árvore na base de teste
y_pred_tunada_valid = arvore_tunada.predict(X_valid)
matriz_confusao_tunada = confusion_matrix(y_valid, y_pred_tunada_valid)

plt.figure(figsize=(8, 6))
sns.heatmap(matriz_confusao_tunada,
            annot=True,
            fmt="d",
            cmap='Oranges',
            xticklabels=nomes_das_classes,
            yticklabels=nomes_das_classes)
plt.title('Matriz de Confusão - Nova Árvore (Base de Teste)')
plt.ylabel('Verdadeiro (Real)')
plt.xlabel('Previsto (Modelo)')
plt.show()

# Observe a distribuição da predição - qual a proporção de 'maus'?
# Como 'mau' = 1 e 'bom' = 0, a média do array de predições nos dá a proporção de 'maus'.
proporcao_maus = np.mean(y_pred_tunada_valid)
print(f"Proporção de proponentes classificados como 'maus' pelo novo modelo: {proporcao_maus*100:.2f}%")

# 5. Como ficaria a acurácia se você classificasse todos os contratos como 'bons'?
print("\n--- Calculando a Acurácia de um Modelo de Linha de Base (Baseline) ---")
# 'Bons' corresponde à classe 0.
# Criada uma previsão onde todos são classificados como 0.
predicao_baseline = np.zeros_like(y_valid)

# Calcula a acurácia dessa previsão "ingênua"
acuracia_baseline = accuracy_score(y_valid, predicao_baseline)

print(f"Se todos os contratos fossem classificados como 'bons' (classe 0), a acurácia seria: {acuracia_baseline*100:.2f}%")
print("Qualquer modelo útil deve ter uma acurácia superior a este valor de linha de base.")


# In[ ]:




