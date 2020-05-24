import retro # Libreria para tratar con videojuegos clasicos
import numpy as np # Para operaciones de matrices y vectores 
import cv2 # Para reduccion de las imagenes (frames)
import neat # Libreria para aplicar el algoritmo NEAT
import pickle

env = retro.make(game='SuperMarioBros-Nes', state='Level1-1', record=True)
oned_image = []

def eval_genomes(genomes, config):

    # Cada genoma es un mario en una poblacion de marios

    for genome_id, genome in genomes: # Un mario
        
        ob = env.reset() # Obtener la primer observacion

        # Tomar una accion random
        # Una accion es un array de 1 y 0s, que representan los botones presionados
        random_action = env.action_space.sample()

        # Obtener las dimensiones de la observacion, que es una matriz 3d 
        # de los frames
        inx,iny,inc = env.observation_space.shape # inc es el color

        # Reducimos el tamaño de la imagen para hacer el preprocesamiento mas rapido
        # El algoritmo no necesita imagenes muy detalladas
        inx = int(inx / 8)
        iny = int(iny / 8)

        # Crear 20 RNNs a partir de este genoma.
        # Despues de varias steps, nos quedaremos con las mejores RNNs
        net = neat.nn.RecurrentNetwork.create(genome,config)

        # El fitness nos indicará que tan lejos llegó mario
        current_max_fitness = 0 
        fitness_current = 0 # rewards que va obteniendo

        frame = 0
        counter = 0 # contar el numero de pasos que toma mario
        
        done = False

        while not done: # Mientras no muera 

            env.render() # Mostrar la imagen de Mario

            frame += 1


            ob = cv2.resize(ob,(inx,iny)) # Convertir la observacion en una matriz
            ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY) # No importa si el color es gris
            ob = np.reshape(ob,(inx,iny)) # Matriz con imagenes grises
            
            # Pasar la matriz a un vector 1D
            oned_image = np.ndarray.flatten(ob) 

            # Output de la Red Neural del frame actual
            neuralnet_output = net.activate(oned_image) 

            # Ejecutar la accion que propuso la red neuronal
            # Se obtiene la observacion actual, un reward, un booleano de muerte e informacion adicional
            ob, rew, done, info = env.step(neuralnet_output) 

            # Aumentar reward
            fitness_current += rew


            if fitness_current > current_max_fitness:
                current_max_fitness = fitness_current
                counter = 0
            else:
                # Contar por cuantos "pasos" mario no se mueve
                counter += 1

            # Si muere o si han pasado 250 frames (mario no avanza)
            if done or counter == 250:
                done = True 
                print(genome_id,fitness_current)
            
            genome.fitness = fitness_current

# Mandar la configuracion
# Se trabaja con los valores por default de la libreria
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config-feedforward')
p = neat.Population(config) # Comenzar la poblacion

# Obtener estadisticas por consola
p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)

# Guardar informacion cada 10 frames
p.add_reporter(neat.Checkpointer(10))

winner = p.run(eval_genomes)



with open('winner.pkl', 'wb') as output:
    pickle.dump(winner, output, 1)