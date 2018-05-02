Para executar o algoritmo cartpole.py programa são necessários ter instalados os seguintes pacotes:
gym
cPickle
numpy
time
random

Para executar o programa simplesmente digite no terminal:

python cartpole.py

O resultado do teste do agente treinado será mostrado no terminal ao final da execução.

As variáveis possíveis de serem alteradas e seus respectivos reflexos sobre a execução:

num_train_episodes: (inteiro)
É a quantidade de episódios que com que um agente é treinado, o artigo expõem que o agente consegue obter
a pontuação máxima entre a quantidade de 4000 e 5000 episódios de treinamento.

num_test_episodes: (inteiro)
Com quantos episódios o agente será testado, note que o resultado final (mostrado no terminal) é a média
da performace destes testes.

render: (True ou False)
Indica se os episódios (cada passo) do teste devem ser renderizados na tela ou não. Se estivere marcado
como verdadeiro (True) a execução do programa irá demorar muito mais.

slow_render: (True ou False)
Desacelera a renderização de cada passo em 0.2 segundos. Não tem efeito se a rederização está marcada
como falsa (render = False)

use_saved_qtable: (True ou False)
Informa se a tabela de estado x ação (qtable) armazenada deve ser usada. Necessário caso haja a necessidade
de não treinar o agente do zero. Se não existir uma tabela salva e esta opção estiver marcada como verdadeira,
o programa informará que não existe e fechará.

print_rewards = (True ou False)
Mostra a recompensa obtida a cada episódio em um teste.

Obs: Existe, ainda, uma tabela qtable chamada "score200.data" que foi treinada com 50000 episódios. Para utiliza-la
nomeie-a como "qtable.data" e deixe a variável use_saved_qtable = True
