from tqdm.auto import tqdm
import numpy as np
import sys
from ddpg import Agent
import matplotlib.pyplot as plt
from environment import Env
from plot import MakePLT

# Arguments:
# Range = Buffer capacity for both data and energy BufferError
# Quick Eval = Load and Evaluate
# Load = Load pretrained model
# Evaluate = Evaluate trained policy
# lam_1, lam_2,..., lam_N = expected data arrival rate for N nodes
# lam_E_1, lam_E_2,..., lam_E_N = expected energy arrival rate for N nodes

if __name__=='__main__':
    arguments = sys.argv[1:]# Range, Eval, Load Evaluate, Lam, LamE
    arguments = np.array(arguments).astype('float')
    rng = int(arguments[0])
    nodes = len(arguments[4:])//2
    lam, lamE = arguments[4:4+nodes], arguments[4+nodes:4+2*nodes]#[0.5,4.5],[5,5]
    #[0.5, 2.5, 2.0, 3.0, 1.0],[5, 5, 5, 5, 5]#
    #[0.5, 2.5, 0.7, 2.0, 1.5, 3.0, 2.8, 3.2, 1.9, 1.0], [5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
    env = Env(rng, nodes, lam, lamE)
    agent = Agent(input_dims=nodes*2, env=env,
            rng=rng, n_actions=nodes**2,
            nodes=nodes, fc1=2, fc2=4)
    n_games, n_steps = 1, 5000#2500, 500#
    N = n_games*n_steps
    l_n = np.radians(np.arange(N))
    cos = np.cos(l_n)
    V = 2
    noise = np.zeros(N)
    for i in range(N):
        F = (N-i)/N
        noise[i] = V*F*(1 + cos[i])/2
    l_str = '_'
    for l in lam:
        l_str = l_str+str(l)+'_'
    l_str_E = '_'
    for l in lamE:
        l_str_E = l_str_E+str(l)+'_'
    added = 'EHWSN_DDPG_'+str(nodes)+'_nodes_'+str(rng)+'lamD'+l_str+'lamE'+l_str_E
    figure_file = 'plots/'+added+'.png'
    array_file = 'store/'+added+'.npy'

    '''wandb.init(
        # Set the project where this run will be logged
        project="EHWSN-DDPG", 
        # We pass a run name (otherwise it will be randomly assigned, like sunshine-lollypop-10)
        name=f"experiment_1_2.0_5.0", 
        # Track hyperparameters and run metadata
        config={
        "Architecture": "DDPG",
        "Range": rng,
        "Lambda": lam,
        "Lambda E": lamE,
        "episodes": n_games,
        "steps": n_steps,
        })'''

    best_score = -10000000000000
    score_history = []
    quick_eval = bool(arguments[1])#True#False#
    load_checkpoint = bool(arguments[2])#False or quick_eval#True#
    evaluate = bool(arguments[3])#False or quick_eval#True#
    History = np.zeros((3*nodes+2, N))#Q,E,Sh,R L

    if load_checkpoint:
        n_steps_check = 0
        while n_steps_check<=agent.batch_size:
            observation = env.reset()
            action = np.random.randint(rng, size=(nodes,nodes))#env.action_space.sample()[0][0]
            observation_, reward, done, info = env.step(action)
            act = action.reshape(nodes**2)
            agent.remember(observation, act, reward, observation_, done)
            n_steps_check +=1
        agent.learn()
        #print(agent.actor.summary())
        #print(agent.critic.summary())
        #print(agent.target_actor.summary())
        #print(agent.target_critic.summary())
        agent.load_models()
        #evaluate=True
    #else:
    #    evaluate=False
    print()
    print("Training Start")
    print()
    for g in tqdm(range(n_games)):
        observation = env.reset()
        done = False
        score, reward = 0, 0
        #for s in range(n_steps):
        for s in tqdm(range(n_steps)):
            idx = s+g*n_games
            action = agent.choose_action(observation, noise[idx], evaluate)
            act = np.array(action).reshape((nodes,nodes))
            observation_, reward, done, loss = env.step(act)

            History[:2*nodes,idx] = observation_
            arr1 = act - np.diag(act)
            for c in range(len(arr1)):
                arr1[c] = arr1[c]/rng
            Sh = np.sum(arr1, axis=1)
            History[2*nodes:3*nodes,idx] = Sh
            History[-2,idx] = reward
            History[-1,idx] = loss

            score += reward
            agent.remember(observation, action, reward, observation_, done)
            if not evaluate:
                agent.learn()
            observation = observation_
            if idx<20:
                continue
            avg_score = np.mean(History[-1,idx-20:idx])
            if avg_score > best_score:
                best_score = avg_score
                if not evaluate:
                    agent.save_models()
                    #wandb.log({"Best": best_score,"Average":avg_score,
                    #"Score History": np.mean(np.array(state_history))})

        score_history.append(score)

        print()
        print('episode ', g, 'score %.1f' % score)#, 'avg score %.1f' % avg_score)
        print()
    
    if not evaluate:
        x = np.arange(1,n_games*n_steps+1)
        Mean = np.zeros(n_games*n_steps)
        for i in range(n_games*n_steps):
            Mean[i] = np.mean(History[-2,:i+1])
        ## REWARD
        #path_fig_R = 'plots/Train/'+added+'R.png'
        #MakePLT(x,Mean, "Steps", "Reward", L=None, path=path_fig_R)
        ## NOISE
        #path_fig_N = 'plots/Train/'+added+'N.png'
        #MakePLT(x,noise, "Steps", "Reward", L=None, path=path_fig_N)
        plt.plot(x, Mean)
        plt.xlabel("Steps")
        plt.xlabel("Reward")
        path_fig = 'plots/Train/'+added+'.png'
        path_arr = 'store/Train/'+added+'.npy'
        plt.savefig(path_fig)
        np.save(path_arr,History)
    else:
        x = np.arange(1,n_games*n_steps+1)#[i+1 for i in range(n_games)]
        Mean = np.zeros((3*nodes+2, n_games*n_steps))
        for i in range(n_games*n_steps):
            Mean[:,i] = np.mean(History[:,:i+1], axis=1)
        print(x.shape)
        print(Mean.shape)
        # QUEUE
        L_Q = ['Q'+str(i+1)+' '+str(lam[i]) for i in range(nodes)]
        path_fig_Q = 'plots/Test/'+added+'Q.png'
        MakePLT(x,Mean[:nodes].T, "Steps", "Queues", L_Q, path_fig_Q)
        #print(np.mean(Mean[:nodes]))
        print("Mean Q",np.mean(Mean[:nodes],axis=1))
        # ENERGY
        L_E = ['E'+str(i+1)+' '+str(lamE[i]) for i in range(nodes)]
        path_fig_E = 'plots/Test/'+added+'E.png'
        MakePLT(x,Mean[nodes:2*nodes].T, "Steps", "Energies", L_E, path_fig_E)
        # SHARED
        L_S = ['S'+str(i+1) for i in range(nodes)]
        path_fig_S = 'plots/Test/'+added+'S.png'
        MakePLT(x,Mean[2*nodes:3*nodes].T, "Steps", "Shares", L_S, path_fig_S)
        # REWARD
        path_fig_R = 'plots/Test/'+added+'R.png'
        MakePLT(x,Mean[-2], "Steps", "Reward", L=None, path=path_fig_R)
        # LOSS
        path_fig_L = 'plots/Test/'+added+'L.png'
        MakePLT(x,Mean[-1], "Steps", "Loss", L=None, path=path_fig_L)
        print("Mean Loss",np.mean(Mean[-1]))

        path_arr = 'store/Test/'+added+'.npy'
        np.save(path_arr,History)