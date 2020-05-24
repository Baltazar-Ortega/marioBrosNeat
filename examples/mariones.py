import retro

env = retro.make(game='SuperMarioBros-Nes', state='Level1-1', record=True)

env.reset()

done = False

while not done:
    env.render()
    
    #action = env.action_space.sample()
    action = [0,0,1,0,0,0,0,1,0,1,0,0]

    
    ob, rew, done, info = env.step(action)
    print("Action ", action, "Reward ", rew)