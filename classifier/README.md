<h1 align="center">
📄<br>Classificação Imagens
</h1>

## Requerimentos
Uma lista de bibliotecas foi utilizada para a criação do código descrito no presente documento, logo, para que se possa utilizar o material apresentado nesse arquivo é necessário instalar todas as bibliotecas previamente a utilização do mesmo. Para isso, existe um arquivo chamado "requirement.txt" na raíz do programa, onde consta todas as bibliotecas utilizadas para execução. Para instalar as bibliotecas pode-se utilizar o Anaconda3, que é um sistema de gerenciamento de projetos em python, em que o mesmo criará um ambiente à partir do arquivo em txt com todos os requerimentos já instalados utilizando o seguinte comando: 

```conda create --name <env> --file requirements.txt```

Ou pode-se utilizar o ambiente do google colaboratory para que não seja necessário instalar nenhum pacote, no entanto, o mesmo não armazenará os dados criados pelo script, nem o modelo neural gerado pelo mesmo.

## Criar dados

Para iniciar a execução do programa é necessário processar as imagens como Vetores binarios e associá-los aos valores de pontos OK e NOK. Para isso, existem dois scripts que executam a tarefa de separá-los e agrupá-los. Para executá-los é necessários utilizar o python3. O primeiro script gera os arquivos necessário para o treinamento:

```python create_data.py```

O segundo script prepara valores para que o treinamento seja validado com os valores treinados previamente:

```python create_data_test.py```

## Treinamento do modelo

Utilizando os valores dos scripts anteriores, que já serão salvos na pasta da aplicação com os nomes a seguir: 'X.pickle, X_test.pickle, y_one.pickle, y_one_test.pickle, y.pickle'; será possível criar e treinar um modelo de redes neurais para a futura avaliação do modelo para imagens novas. Para isso, é necessário utilizar o python3 para a execução do script. Vale lembrar que nesse momento, é essencial utilizar uma versão do TensorFlow compatível com a placa de vídeo do computador e para melhores tempos de execução é essencial ter uma placa de vídeo com no mínimo 4 Gb de memória de vídeo. Para executar o script, rode o comando: 

```python create_model.py```

## Utilizando o modelo

O modelo criado e treinado com o script anterior será salvo na pasta 'models' com o nome 'mymodel'. Para fazer a avaliação de precisão do modelo com os valores de teste indicados, é necessário utilizar um outro script que é responsável por carregar o modelo previamente gerado e também os valores de teste do modelo para que possa ser avaliado, logo, as etapas anteriores precisam ser executadas para que se possa utilizar o script de avaliação. Para utilizá-lo:

```python evaluate_model.py```

## Otimizando o modelo

No arquivo 'create_model.py' existem variáveis que controlam a criação da rede neural do aplicativo, as variáveis são: 'dense_layers, layer_sizes, conv_layers, max_pooling, res, dropouts e drop_rates. As variáveis controlam os números de camadas densas utilizadas, tamanho das camadas, quantidade de camadas convolucionais, máximo de união dos nódulos, resolução das células, número de remoções de características e porcentagem de remoção das características. Quando alterados, os valores gerarão redes diferentes que podem ter impacto na precisão do modelo gerado. Para automatizar a tomada de decisões para o modelo, outro script chamado 'improve_model.py' foi criado, o mesmo realizará diversos testes com essas variáveis apresentadas, para o conjunto de dados fornecido na pasta IMG -> OK -> NOK. Para isso, no script improve_model.py é necessário adicionar diversos valores que se deseja avaliar nas variáveis acima, porém, diferentemente do script de criação do modelo, deve-se adicionar múltiplos valores aos vetores, pois serão testados individualmente com os valores das outras variáveis. Logo, o script acima leva um bom tempo para finalizar a execução e é imprescindível a utilização de uma boa placa de vídeo. Após a execução do script, logs serão gerados na pasta 'logs' onde poderão ser comparados e então escolhidos os melhores valores para a rede neural através da utilização do TensorBoard, onde podem ser carregados esses logs e então avaliadas as precisões de cada rede neural criada.
