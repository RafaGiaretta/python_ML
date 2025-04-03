from numpy import *

class LinearRegression:   #  Criar classe (Inicio dela)
    def __init__(self, x, y): # Metodo para iniciar ( os 2 '_' é uma convencao de que o metodo é privado), self é a representacao da instancia da classe
      # Atributos da classe
      self.x = x
      self.y = y
      self.__correlation_coefficient = self.__correlacao() # Metodos para cada valor
      self.__inclination = self.__inclinacao() # Metodos para cada valor
      self.__intercept = self.__interceptacao() # Metodos para cada valor

    def __correlacao(self): # Metodo de correlacao
      covariacao = cov(self.x, self.y, bias=True) [0][1] # Metodo  cov() faz o calculo da covariacao, bias = true coloca -1 no denominador, o default é false
      variancia_x = var(self.x) # Metodo var() para calcular a variancia de cada 
      variancia_y = var(self.y) # Metodo var() para calcular a variancia de cada 
      return covariacao / sqrt(variancia_x * variancia_y) # Correlacao =  covariação / raiz quadrada de (variancia de x * variancia de y) funcao sqrt para expoente

    def __inclinacao(self): # Metodo de inclinação
      stdx = std(self.x)    # Metodo std() calcula o desvio padrao do valor x 
      stdy = std(self.y)    # Metodo std() calcula o desvio padrao do valor y
      return self.__correlation_coefficient * (stdy / stdx) # Inclinação = correlação * (desvio padrao de y / desvio padrao de x)

    def __interceptacao(self): # Metodo de interceptação
      mediax = mean(self.x)    # Metodo mean() funcao para media de x
      mediay = mean(self.y)    # Metodo mean() funcao para media de y
      return mediay - (self.__inclination * mediax) # Interceptação = media de y - (inclinação * media de x)

    def previsao(self,valor): 
      return self.__intercept + (self.__inclination * valor) # Previsao é feita coma interceptação + (inclinação * valor passado como parametro)
    
                        # Final da Classe
                        
# Exemplo simples
                      
x = array ([1,2,3,4,5]) # Variavel independente
y = array([2,4,6,8,10]) # Variavel dependente

lr = LinearRegression(x,y) # Instanciar a classe com os arrays

previsao = lr.previsao(6) # Chamar o metodo de previsao da classe 
print(previsao)           # Printa o resultado da previsao