# Descobrir quais atributos sao melhores para prever o consumo de um carro
# Dados utilizados na pasta 'Arquivos Utilizados' -> arquivo 'mt_Cars.csv'

import pandas as pd # Biblioteca para manipulação e analise de dados
import numpy as np # Biblioteca para calculos numericos e manipulação de arrays
import matplotlib.pyplot as plt # Biblioteca para visualizacao de dados em Python que permite criar graficos, histogramas..
import statsmodels.formula.api as sm # Bilioteca usada para modelagem estatística e econometria, permite realizar regressao linear, testes estat. e analise de series temporais
import scipy.stats as stats # Biblioteca focada em calculos cientificos e estatísticos, incluindo distribuicoes de probab. e otimizacao
import seaborn as sns # Biblioteca baseada no Matplotlib, mas com graficos mais bonitos..

# Ler o arquivo
base = pd.read_csv('Exercicios\RL com StatsModels.py') # Base vai ser a variavel que le o arquivo, pd.read_csv é a funcao do pandas para ler arquivos csv com endereco como parametro
print(base.shape) # para verificar linhas e colunas do arquivo
base = base.drop(['Unnamed: 0'], axis=1) # Removemos a coluna de nomes para nao ter que ficar tratando o que nao é numero
# print(base.head()) # possibilita visualiar em tabela o arquivo

# Gerando um Correlograma ( Forma de criar uma matriz que mostre a correlação entre todos os dados da tabela )
corr = base.corr() # Gerando um objeto que vai ter uma matriz destas correlacoes
sns.heatmap(corr, cmap='coolwarm', annot=True, fmt='.2f') # Chama a biblioteca sns para criar num grafico(variavel de correlação, tipo de grafico, annot mostra os valores em cada celula, fmt é formatação dos valores)
# plt.show() # Mostra o grafico gerado


# Gerar um grafico de dispersao para avaliar as mais provaveis das variaveis

column_pairs =[('mpg','cyl'),('mpg','disp'),('mpg','hp'),('mpg','drat'),('mpg','wt'),('mpg','vs')] # Tuplas para comparacao
n_plots = len(column_pairs) # Variavel para calcular o numero de pares
fig, axes = plt.subplots(nrows =n_plots, ncols=1, figsize=(6,4 * n_plots)) # Figs sao figuras e axes eixos que vao ser retornados pelo metodo subplot(numero de linhas = n_plots, num de colunas, e tamanho da figura)

# Percorrer estes pares de colunas para gerar os graficos

for i, pair in enumerate(column_pairs): # Laco para percorrer cada par
    x_col, y_col = pair # Definindo cara par
    sns.scatterplot(x=x_col, y=y_col, data=base, ax=axes[i]) # Cria grafico de dispercao
    axes[i].set_title(f'{x_col} vs {y_col}') # Define o titulo do subgrafico
    
plt.tight_layout() # ajustar layout
plt.show()

# Neste modelo AIC 156.6 e  BIC 162.5 Shapiro-Wilk statística: 0.927, p-value: 0.033
# modelo = sm.ols(formula='mpg ~ wt + disp + hp', data=base) # Biblioteca de modelos statsmodels.ols(formula) se separa variaveis dependente das independentes com um sinal de '~' Como criar o modelo

# Neste modelo AIC 165.1 e  BIC 169.5 Shapiro-Wilk statística: 0.942, p-value: 0.085
# modelo = sm.ols(formula='mpg ~  disp + cyl', data=base)

# Neste modelo AIC 179.1 e  BIC 183.5 Shapiro-Wilk statística: 0.981, p-value: 0.822 <- modelo bem a cima do valor de parametro de comparacao e nao e uma hipotese nula 
modelo = sm.ols(formula='mpg ~  drat + vs', data=base)
modelo = modelo.fit() # Fit cria de fato o modelo
modelo.summary() # Cria um sumario


# AIC E BIC é metrica de performace, p-value mede a distribuicao de residuos
# Para escolher o melhor modelo vai levar estas caracteristicas, alem de depender da analise e interpretação do analista
print(modelo.summary())

# Residuos
residuos = modelo.resid
plt.hist(residuos, bins=20)
plt.xlabel("Residuos")
plt.ylabel("Frequencia")
plt.title("Histogama de Residuos")
plt.show()

stats.probplot(residuos, dist="norm", plot=plt)
plt.title("Q-Q Plot de Residuos")
plt.show()


# Teste de Shapiro wilk

stat, pval = stats.shapiro(residuos) # Teste estatistico
print(f'Shapiro-Wilk statística: {stat:.3f}, p-value: {pval:.3f}')

#Shapiro-Wilk statística: 0.927, p-value: 0.033
# H0 = hipotese nulan os dados estao normalmente distribuidos p <= 0.05 rejeito a  hipotese nula ( nao estao normalmente distribuidos )
# p > 0.05 não é possiveil rejeitar a hipotese nula
# Statistica quanto mais perto de 1 melhor

