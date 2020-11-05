from ResAlloc_env import *
from trainer import *
from NN import *
from replay import ReplayMemory
from mcts import *
import matplotlib.pyplot as plt

import random

if __name__ == '__main__':

    def reverse_input(boof):
        wow = []
        for row in boof:
            wow.append(np.sum(row))
        return wow

    def prep_input(state, env):

        return np.array(state).reshape(-1,1)

        #one hot
        r = []
        for i in range(len(state)):
            if state[i] > 0:
                r.append(i)
        inp = np.zeros((env.n_nodes, env.n_resources))
        for i in range(len(r)):
            inp[r[i]][i] = 1.0
        # inp = np.hstack((inp, np.zeros((inp.shape[0], 1), dtype=inp.dtype)))
        # for i in range(inp.shape[0]):
        #     inp[i][-1] = self.TreeEnv.P_val[i]
        return inp

    env = gameEnv(0)
    nPass = 32
    stsW = 1
    representation = Representation(lambda: RepresentationFunc(env.features.shape[1] + stsW, [64, 64], nPass))

    trainer_board = Trainer(lambda: PredictionNN(nPass, 64, 64, env.n_resources + 1, 0.3), representation, env, 'board')
    trainer_node = []
    for i in range(env.n_nodes):
        trainer_node.append(Trainer(lambda: PredictionNN(nPass, 64, 64, env.degree[i], 0.3), representation, env, 'node'))
    trainer_score = ScorePredictionTrainer(lambda: ScorePredictionNN(nPass, 64, 64, 0.3), representation, env)

    # trainer_board = Trainer(lambda: GCNBoard(4, 16, env.n_resources, env.n_nodes, 0.2), representation, env, 'board')
    # trainer_node = []
    # for i in range(env.n_nodes):
    #     trainer_node.append(Trainer(lambda: GCNNode(4, 16, env.degree[i], env.n_nodes, 0.2, 'node'+str(i)), representation, env, 'node'))
    # trainer_score = ScorePredictionTrainer(lambda: ScorePredictionFunc(5, 10, 10, env.n_nodes, 0.2), representation, env)


    mem_scores = ReplayMemory(500, {"sts" : [env.n_nodes, stsW], "features" : [env.features.shape[0], env.features.shape[1]], "scores" : []})
    mem_board = ReplayMemory(100, {"sts" : [env.n_nodes, stsW], "features" : [env.features.shape[0], env.features.shape[1]], "pi" : [env.n_resources+1], "return" : []})
    mem_node = []
    for i in range(env.n_nodes):
        mem_node.append(ReplayMemory(100, {"sts" : [env.n_nodes, stsW], "features" : [env.features.shape[0], env.features.shape[1]], "pi" : [env.degree[i]], "return" : []}))

    # mem_scores = ReplayMemory(500, {"sts" : [env.n_nodes, 1], "features" : [env.features.shape[0], env.features.shape[1]], "scores" : []}, batch_size = 10)
    # mem_board = ReplayMemory(100, {"sts" : [env.n_nodes, 1], "features" : [env.features.shape[0], env.features.shape[1]], "pi" : [env.n_resources+1], "return" : []})
    # mem_node = []
    # for i in range(env.n_nodes):
    #     mem_node.append(ReplayMemory(100, {"sts" : [env.n_nodes, 1], "features" : [env.features.shape[0], env.features.shape[1]], "pi" : [env.degree[i]], "return" : []}))

    lr = 1.0

    res1 = []
    res2 = []
    for i in range(10):
        res1.append([])
        res2.append([])


    # for i in range(15):
    #     print("\nfinalee ", i)
    #     env = gameEnv(i)
    #
    #     sts_board, searches_pi_board, ret_board, sts_node, searches_pi_node, ret_node, score_sts, scores = execute_episode(trainer_board, trainer_node, trainer_score, representation, 800, env, 'train', 1.0)
    #
    #     for i in range(len(sts_board)):
    #         wow = reverse_input(sts_board[i])
    #         print(i, wow, np.sum(wow*env.out))

    gauu = [0.0]*(env.n_resources+1)
    # for i in range(40):
    #     score_sts = []
    #     scores = []
    #
    #     for wow in range(50):
    #         scene = random.randint(0, 9)
    #         env = gameEnv(scene)
    #         for num in range(1):
    #             koyta = random.randint(0,env.n_resources)
    #             triggered = []
    #             not_triggered = []
    #             for j in range(env.n_nodes):
    #                 if env.out[j] == 1.0:
    #                     triggered.append(j)
    #                 else:
    #                     not_triggered.append(j)
    #
    #             sts = [0.0]*env.n_nodes
    #             jadu = random.sample(triggered, min(koyta, len(triggered)))
    #             for idx in jadu:
    #                 sts[idx] = 1.0
    #             # print(env.n_nodes, triggered, not_triggered, env.n_resources, min(koyta, len(triggered)))
    #             jadu = random.sample(not_triggered, env.n_resources - min(koyta, len(triggered)))
    #             for idx in jadu:
    #                 sts[idx] = 1.0
    #
    #             sts = prep_input(sts, env)
    #             score_sts.append(sts)
    #             scores.append(koyta/env.n_resources)
    #             gauu[koyta] += 1.0
    #             mem_scores.add_all({"sts" : score_sts, "features": [env.features]*len(score_sts), "scores": scores})
    #
    #     if mem_scores.count>= 10:
    #         # print("sc ", mem_scores.count)
    #         batch_score = mem_scores.get_minibatch()
    #         # print(batch_score["sts"], batch_score["features"], batch_score["scores"])
    #         loss = trainer_score.train(batch_score["sts"], batch_score["features"], batch_score["scores"])


    print(trainer_score.step_model.parameters())
    for i in range(100):
        if (i+1)%10==0:
            lr = lr + 1.0
        scenario = random.randint(0, 9)
        print("\n\nayayayaya -----", i, scenario)
        env = gameEnv(scenario)

        sts_board, searches_pi_board, ret_board, sts_node, searches_pi_node, ret_node, score_sts, scores = execute_episode(trainer_board, trainer_node, trainer_score, representation, 800, env, 'train', 1.0)

        wow = reverse_input(sts_board[-1])
        print(len(sts_board), wow, np.sum(wow*env.out), np.sum(env.out))
        res1[scenario].append(len(sts_board))
        res2[scenario].append(np.sum(wow*env.out))

        mem_board.add_all({"sts" : sts_board,"features": [env.features]*len(sts_board),  "pi" : searches_pi_board, "return" : ret_board})
        for i in range(env.n_nodes):
            mem_node[i].add_all({"sts" : sts_node[i],"features": [env.features]*len(sts_node[i]), "pi" : searches_pi_node[i], "return" : ret_node[i]})

        if mem_board.count >= 16:
            trainer_board.learning_rate = 0.1/float(lr)
            # print("bt",mem_board.count)
            batch_board = mem_board.get_minibatch()
            #print(batch_board["sts"], batch_board["pi"], batch_board["return"])
            loss = trainer_board.train(batch_board["sts"], batch_board["features"], batch_board["pi"], batch_board["return"])
            #print("pipilika", loss.item())

        for i in range(env.n_nodes):
            # print("nd",i, mem_node[i].count)
            if mem_node[i].count >= 8:
                trainer_node[i].learning_rate = 0.2/float(lr)
                batch_node = mem_node[i].get_minibatch()
                loss = trainer_node[i].train(batch_node["sts"], batch_node["features"], batch_node["pi"], batch_node["return"])

    # point_names = {}
    # for i in range(10):
    #     if len(res1[i]) == 0:
    #         continue
    #     point_names.clear()
    #     print(i, len(res1[i]), res1[i], res2[i])
    #     plt.plot(res1[i], res2[i], 'ro')
    #     for j in range(len(res1[i])):
    #         if (res1[i][j], res2[i][j]) in point_names:
    #             point_names[(res1[i][j], res2[i][j])] = point_names[(res1[i][j], res2[i][j])] + "," + str(j)
    #         else:
    #             point_names[(res1[i][j], res2[i][j])] = str(j)
    #     for item in point_names.items():
    #         plt.annotate(item[1], item[0])
    #     plt.axis([0, 15, 0, 4])
    #     plt.xlabel('#moves')
    #     plt.ylabel('score')
    #     plt.show()
    #     plt.clf

    for i in range(5):
        print("\nfinalee ", 10+i)
        env = gameEnv(10+i)

        sts_board, searches_pi_board, ret_board, sts_node, searches_pi_node, ret_node, score_sts, scores = execute_episode(trainer_board, trainer_node, trainer_score, representation, 1000, env, 'test', 1.0)

        for i in range(len(sts_board)):
            wow = reverse_input(sts_board[i])
            print(i, wow, np.sum(wow*env.out))

        sts_board, searches_pi_board, ret_board, sts_node, searches_pi_node, ret_node, score_sts, scores = execute_episode(trainer_board, trainer_node, trainer_score, representation, 1000, env, 'train', 1.0)

        for i in range(len(sts_board)):
            wow = reverse_input(sts_board[i])
            print(i, wow, np.sum(wow*env.out))
