from ResAlloc_env import *
from trainer import *
from NN import *
from replay import ReplayMemory
from mcts import *

if __name__ == '__main__':
    nodes = 10
    resources = 3

    env = gameEnv()

    trainer_board = Trainer(lambda: GCNBoard(env.n_resources+1, 8, env.n_resources, env.n_nodes, 0.2), env, 'board')
    trainer_node = []
    for i in range(nodes):
        trainer_node.append(Trainer(lambda: GCNNode(env.n_resources+1, 8, env.degree[i], env.n_nodes, 0.2, 'node'+str(i)), env, 'node'))

    mem_board = ReplayMemory(100, {"sts" : [env.adj.shape[0]], "pi" : [env.n_resources+1], "return" : []})
    mem_node = []
    for i in range(env.adj.shape[0]):
        mem_node.append(ReplayMemory(100, {"sts" : [env.adj.shape[0]], "pi" : [env.degree[i]], "return" : []}))

    lr = 1.0


    for i in range(10):
        if (i+1)%5==0:
            lr = lr + 1.0

        print("ayayaya ----", i)
        # if i%3 == 0:
        #     sts_board, searches_pi_board, ret_board, sts_node, searches_pi_node, ret_node = execute_episode(trainer_board, trainer_node, 700, env, 1.0)
        # else:
        #     sts_board, searches_pi_board, ret_board, sts_node, searches_pi_node, ret_node = execute_episode(trainer_board, trainer_node, 700, env, 1.0)

        sts_board, searches_pi_board, ret_board, sts_node, searches_pi_node, ret_node = execute_episode(trainer_board, trainer_node, 1200, env, 1.0)

        for i in range(len(sts_board)):
            # print(i, sts_board[i], np.sum(sts_board[i]*env.P_val))
            print(i, sts_board[i], np.sum(sts_board[i]*env.out))

        mem_board.add_all({"sts" : sts_board, "pi" : searches_pi_board, "return" : ret_board})
        for i in range(env.adj.shape[0]):
            mem_node[i].add_all({"sts" : sts_node[i], "pi" : searches_pi_node[i], "return" : ret_node[i]})

        if mem_board.count >= 8:
            trainer_board.learning_rate = 0.06/float(lr)
            # print("bt",mem_board.count)
            batch_board = mem_board.get_minibatch()
            #print(batch_board["sts"], batch_board["pi"], batch_board["return"])
            loss = trainer_board.train(batch_board["sts"], batch_board["pi"], batch_board["return"])
            #print("pipilika", loss.item())

        for i in range(env.adj.shape[0]):
            # print("nd",i, mem_node[i].count)
            if mem_node[i].count >= 2:
                trainer_node[i].learning_rate = 0.06/float(lr)
                batch_node = mem_node[i].get_minibatch()
                loss = trainer_node[i].train(batch_node["sts"], batch_node["pi"], batch_node["return"])

    for i in range(40):
        if (i+1)%5==0:
            lr = lr + 1.0

        sts_board, searches_pi_board, ret_board, sts_node, searches_pi_node, ret_node = execute_episode(trainer_board, trainer_node, 1200, env, 1.0)

        for i in range(len(sts_board)):
            # print(i, sts_board[i], np.sum(sts_board[i]*env.P_val))
            print(i, sts_board[i], np.sum(sts_board[i]*env.out))

        mem_board.add_all({"sts" : sts_board, "pi" : searches_pi_board, "return" : ret_board})
        for i in range(env.adj.shape[0]):
            mem_node[i].add_all({"sts" : sts_node[i], "pi" : searches_pi_node[i], "return" : ret_node[i]})

        if mem_board.count >= 8:
            trainer_board.learning_rate = 0.03/float(lr)
            # print("bt",mem_board.count)
            batch_board = mem_board.get_minibatch()
            #print(batch_board["sts"], batch_board["pi"], batch_board["return"])
            loss = trainer_board.train(batch_board["sts"], batch_board["pi"], batch_board["return"])
            #print("pipilika", loss.item())

        for i in range(env.adj.shape[0]):
            # print("nd",i, mem_node[i].count)
            if mem_node[i].count >= 2:
                trainer_node[i].learning_rate = 0.06/float(lr)
                batch_node = mem_node[i].get_minibatch()
                loss = trainer_node[i].train(batch_node["sts"], batch_node["pi"], batch_node["return"])
