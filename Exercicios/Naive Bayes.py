import pandas as pd

from sklearn.model_selection import train_test_split # Para dividir nossos dados em training test
from sklearn.naive_bayes import GaussianNB # Para utilizar este tipo de teste
from sklearn.preprocessing import LabelEncoder # Para converter dados categoricos em numericos
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report # para avaliar as metricas de desempenho do modelo
from yellowbrick.classifier import ConfusionMatrix # Para gerar visualmente uma matriz de confusao


base = pd.read_csv('\\Exercicios\\Arquivos_Utilizados\\insurance.csv')
base= base.dropna() # Removendo os campos NaN
base = base.drop(columns=['Unnamed: 0'])
print(base)


# Separar a variaveis dependente das demais variaveis
y = base.iloc[:,7].values # Metodo do pandas para pegar colunas - Separando a coluna 7 
X = base.iloc[:,[0,1,2,3,4,5,6,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26]].values

labelEncoder = LabelEncoder() # Instanciando um objeto LabelEncoder


for i in range(X.shape[1]): # Um laco que percorre o comprimento ou numero de colunas do objeto metodo 'shape' mostra numero de linhas e colunas e com o [1] pegamos apenas o valor de colunas
    if X[:,i].dtype == 'object': # Se a variavel for categorica (Texto e nao numero)
        X[:,i] = labelEncoder.fit_transform(X[:,i]) # Pega os dados e tranforma em dados numericos ( Sintaxe X[':'significa todos os dados, i = posição])


print (X)

# Dividir os dados em treino e teste

X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(X,y,test_size=0.3,random_state=1) # Variaveis de treinamento, metodo train_test_split 
                                                        #(Variavel X, variavel y, tamanho do teste (30% vai para teste, o restante (70%) pra treinamento), random_state divide repetindo a aleatoriedade)   
                                                        #(X variavel independente, y variavel dependente)

# Iniciar modelo apos treinamento

modelo = GaussianNB() # Metodo de treinamento
modelo.fit(X_treinamento, y_treinamento) # Fornecendo as variaveis treinadas

# AIC 179.1 BIC 183.5
previsoes = modelo.predict(X_teste) # Para previsoes nao é necessario passar y


accuracy = accuracy_score(y_teste, previsoes)
#print(accuracy) # 0.838 -> quanto mais perto do 1 melhor

# Outras metricas 

precision = precision_score(y_teste, previsoes, average='weighted')
recall = recall_score(y_teste, previsoes, average='weighted')
f1 = f1_score(y_teste, previsoes, average='weighted')

print(f'Acuracia: {accuracy}, Precisao: {precision}, Recall: {recall}, F1: {f1}')

report = classification_report(y_teste, previsoes)

print (report)

confusao = ConfusionMatrix(modelo, classes=['None','Severe','Moderate', 'Mild'])
confusao.fit(X_treinamento,y_treinamento)
confusao.score(X_teste,y_teste)
confusao.poof
confusao.show()