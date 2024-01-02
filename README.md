# Projeto de Machine Learning: Avaliação de fadiga mental de pessoas funcionárias

O projeto demonstra como usar o modelo para fazer predições sobre fadiga mental de pessoas colaboradoras, no intuito de enviar possíveis desgastes e deteriorização do bem-estar destas pessoas.

## Funcionalidades:

Carregamento dos Dados:

- Importação de  as bibliotecas necessárias:

from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

- Download dos arquivos (train e test) .csv contendo o dados de treinamento e teste.

Pré-processamento dos Dados:

- Seleção dos atributos relevantes, como designação e  alocação de recursos.
- Realiza o tratamento de dados ausentes ou inconsistentes.
-  Faz a aivisão em Conjuntos de Treinamento e Teste:
- Utiliza a biblioteca scikit-learn para dividir os dados em conjuntos de treinamento e teste.

Treinamento do Modelo de Regressão Linear:

Inicializar e treinar um modelo de regressão linear utilizando o conjunto de treinamento.

Avaliação do Modelo:

- Calcula a métrica de Raiz Quadrada do Erro Quadrático Médio (RMSE) para avaliar a precisão do modelo.
- Visualiza gráficos de dispersão para comparar as predições com os valores reais.

Predição de Fadiga Mental:

- Utiliza o modelo treinado para fazer predições sobre a fadiga mental com base em novos dados.

## Como Usar

- Faça o download do arquivo.ipynb: se você ainda não fez isso, baixe o arquivo do projeto, geralmente com a extensão.ipynb, para o seu computador.
- Abra o Google Colab.
- Carregue o arquivo.ipynb: no menu "Arquivo" do Google Colab, selecione a opção "Fazer upload de notebook". Em seguida, escolha o arquivo.ipynb que você baixou no passo anterior.

- Importe as bibliotecas necessárias: No início do notebook, as bibliotecas necessárias já devem estar importadas. Verifique se as importações são as seguintes:

import numpy as np;
import pandas as pd;
from sklearn.tree import DecisionTreeClassifier;
from sklearn.model_selection import train_test_split;
from sklearn.metrics import mean_squared_error, accuracy_score;
from sklearn import tree;
import graphviz

- Execute o notebook: Clique em "Runtime" no menu superior e selecione "Run All". Isso executará todas as células do notebook, incluindo o download dos dados, a divisão dos dados, o treinamento do modelo e a avaliação do modelo.



## Observações

- Certifique-se de que todas as bibliotecas necessárias estão instaladas no ambiente do Google Colab.
- Ao executar o notebook, observe as mensagens de saída para garantir que cada etapa seja concluída com êxito.
- Os resultados, métricas de avaliação e visualizações do modelo serão apresentados ao final da execução do notebook.
- Explore os comentários documentação interna e textos no notebook para uma compreensão mais detalhada de cada etapa do projeto.


# Machine Learning project: Assessing mental fatigue in employees

The project demonstrates how to use the model to make predictions about mental fatigue in employees, in order to send possible wear and tear and deterioration in the well-being of these people.

## Functionalities:

Data loading:

- Import of the necessary libraries:

from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

- Download the .csv files (train and test) containing the training and test data.

Data pre-processing:

- Selection of relevant attributes, such as designation and resource allocation.
- Handles missing or inconsistent data.
- Splits into Training and Test Sets:
- Uses the scikit-learn library to split the data into training and test sets.

Linear Regression Model Training:

Initialize and train a linear regression model using the training set.

Model Evaluation:

- Calculates the Root Mean Square Error (RMSE) metric to evaluate the model's accuracy.
- Displays scatter plots to compare predictions with actual values.

Mental Fatigue Prediction:

- Uses the trained model to make predictions about mental fatigue based on new data.

## How to use

- Download the.ipynb file: if you haven't already done so, download the project file, usually with the extension.ipynb, to your computer.
- Open Google Colab.
- Upload the.ipynb file: in Google Colab's "File" menu, select the "Upload from notebook" option. Then choose the.ipynb file you downloaded in the previous step.

- Import the necessary libraries: At the start of the notebook, the necessary libraries should already be imported. Check that the imports are as follows:

import numpy as np;
import pandas as pd;
from sklearn.tree import DecisionTreeClassifier;
from sklearn.model_selection import train_test_split;
from sklearn.metrics import mean_squared_error, accuracy_score;
from sklearn import tree;
import graphviz

- Run the notebook: Click on "Runtime" in the top menu and select "Run All". This will run all the cells in the notebook, including downloading the data, splitting the data, training the model and evaluating the model.

## Notes

- Make sure that all the necessary libraries are installed in the Google Colab environment.
- When running the notebook, watch the output messages to ensure that each step is completed successfully.
- The results, evaluation metrics and visualizations of the model will be presented at the end of the notebook run.
- Explore the comments, internal documentation and texts in the notebook for a more detailed understanding of each stage of the project.
