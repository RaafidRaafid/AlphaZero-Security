
import math
import random as rd
import collections
import numpy as np

# Exploration constant
# c_PUCT = 1.38
c_PUCT = 1.72
# c_PUCT = 2.76
# Dirichlet noise alpha parameter.
D_NOISE_ALPHA = 0.1
# Number of steps into the episode after which we always select the
# action with highest action probability rather than selecting randomly
TEMP_THRESHOLD = 4
STOP_CUTTER = 0.0
mode = 'train'


class DummyNode:
    """
    Special node that is used as the node above the initial root node to
    prevent having to deal with special cases when traversing the tree.
    """

    def __init__(self):
        self.parent = None
        self.child_N = collections.defaultdict(float)
        self.child_W = collections.defaultdict(float)

    def revert_virtual_loss(self, up_to=None): pass

    def add_virtual_loss(self, up_to=None): pass

    def revert_visits(self, up_to=None): pass

    def backup_value(self, value, up_to=None): pass


class MCTSNode:
    """
    Represents a node in the Monte-Carlo search tree. Each node holds a single
    environment state.
    """

    map = {}
    count = 0.0

    def __init__(self, state, n_actions, TreeEnv, type, action=None, parent=None):
        """
        :param state: State that the node should hold.
        :param n_actions: actions that can be performed in each
        state. Equal to the number of outgoing edges of the node.
        :param TreeEnv: Static class that defines the environment dynamics,
        e.g. which state follows from another state when performing an action.
        :param action: Index of the action that led from the parent node to
        this node.
        :param parent: Parent node.
        """
        self.TreeEnv = TreeEnv
        if parent is None:
            self.depth = 0
            parent = DummyNode()
        else:
            self.depth = parent.depth+1
        self.type = type
        self.parent = parent
        self.action = action
        self.state = state
        self.n_actions = n_actions
        self.is_expanded = False
        self.is_stopper = False
        self.n_vlosses = 0  # Number of virtual losses on this node
        self.child_N = np.zeros([len(n_actions)], dtype=np.float32)
        self.child_W = np.zeros([len(n_actions)], dtype=np.float32)
        # Save copy of original prior before it gets mutated by dirichlet noise
        self.original_prior = np.zeros([len(n_actions)], dtype=np.float32)
        self.child_prior = np.zeros([len(n_actions)], dtype=np.float32)
        #print(self.child_prior)
        self.children = {}
        self.N = 0.0
        self.W = 0.0

        self.action_idx = {}
        for i in range(len(n_actions)):
            self.action_idx[n_actions[i]] = i
        self.idx_action = {}
        for i in range(len(n_actions)):
            self.idx_action[i] = n_actions[i]

        self.bad = np.ones([len(n_actions)], dtype=np.float32)

    @property
    def Q(self):
        """
        Returns the current action value of the node.
        """
        return self.W / (1 + self.N)

    @property
    def child_Q(self):
        for i in range(len(self.n_actions)):
            if self.n_actions[i] in self.children:
                self.child_N[i] = self.children[self.n_actions[i]].N
                self.child_W[i] = self.children[self.n_actions[i]].W
        return self.child_W / (1 + self.child_N)

    @property
    def child_U(self):
        #print("-", self.type, self.child_prior, self.child_N, self.state, self)
        for i in range(len(self.n_actions)):
            if self.n_actions[i] in self.children:
                self.child_N[i] = self.children[self.n_actions[i]].N
        return (c_PUCT * math.sqrt(1 + self.N) *
                self.child_prior / (1 + self.child_N))

    @property
    def child_action_score(self):
        """
        Action_Score(s, a) = Q(s, a) + U(s, a) as in paper. A high value
        means the node should be traversed.
        """
        # print(self.child_Q, self.child_U)
        return self.bad*(self.child_Q + self.child_U)

    def select_leaf(self):
        """
        Traverses the MCT rooted in the current node until it finds a leaf
        (i.e. a node that only exists in its parent node in terms of its
        child_N and child_W values but not as a dedicated node in the parent's
        children-mapping). Nodes are selected according to child_action_score.
        It expands the leaf by adding a dedicated MCTSNode. Note that the
        estimated value and prior probabilities still have to be set with
        `incorporate_estimates` afterwards.
        :return: Expanded leaf MCTSNode.
        """
        current = self
        while True:
            current.N += 1
            # Encountered leaf node (i.e. node that is not yet expanded).
            if not current.is_expanded:
                break
            if current.is_done() or current.is_stopper:
                break
            # Choose action with highest score.
            best_move = np.argmax(current.bad*current.child_action_score)
            current = current.maybe_add_child(best_move)
        return current

    def maybe_add_child(node, action):
        """
        Adds a child node for the given action if it does not yet exists, and
        returns it.
        :param action: Action to take in current state which leads to desired
        child node.
        :return: Child MCTSNode.
        """
        action = node.idx_action[action]

        if action not in node.children:

            new_state = node.TreeEnv.next_state(node.state, action, node.type)

            if node.type == 'board':
                if action == -1:
                    node.children[action] = MCTSNode(new_state, [], node.TreeEnv, type = 'board', action = node.action_idx[action], parent=node)
                    # urgent, what is this?
                    node.children[action].is_stopper = True
                else:
                    node.children[action] = MCTSNode(new_state, node.TreeEnv.actions[action], node.TreeEnv, type = 'node', action=node.action_idx[action], parent=node)
            else:
                temp = [0.0]*len(new_state)
                temp[:] = new_state[:]
                ttemp = []
                ttemp.append(-1)
                for i in range(len(temp)):
                    if temp[i] > 0:
                        ttemp.append(i)
                #print(temp, action, action[0], self.state, ttemp)
                node.children[action] = MCTSNode(new_state, ttemp,
                                                node.TreeEnv, type = 'board',
                                                action=node.action_idx[action], parent=node)
                if (str(node.state),str(new_state)) in MCTSNode.map:
                    MCTSNode.map[(str(node.state),str(new_state))] += 1.0
                else:
                    MCTSNode.map[(str(node.state),str(new_state))] = 1.0

                MCTSNode.count += 1.0
            #print(">>>", self.children[action].type, self.children[action].state, self.children[action])
        return node.children[action]

    def add_virtual_loss(self, up_to):
        """
        Propagate a virtual loss up to a given node.
        :param up_to: The node to propagate until.
        """
        self.n_vlosses += 1
        self.W -= 1
        if self.parent is None or self is up_to:
            return
        self.parent.add_virtual_loss(up_to)

    def revert_virtual_loss(self, up_to):
        """
        Undo adding virtual loss.
        :param up_to: The node to propagate until.
        """
        self.n_vlosses -= 1
        self.W += 1
        if self.parent is None or self is up_to:
            return
        self.parent.revert_virtual_loss(up_to)

    def revert_visits(self, up_to):
        """
        Revert visit increments.
        Sometimes, repeated calls to select_leaf return the same node.
        This is rare and we're okay with the wasted computation to evaluate
        the position multiple times by the dual_net. But select_leaf has the
        side effect of incrementing visit counts. Since we want the value to
        only count once for the repeatedly selected node, we also have to
        revert the incremented visit counts.
        :param up_to: The node to propagate until.
        """
        self.N -= 1
        if self.parent is None or self is up_to:
            return
        self.parent.revert_visits(up_to)

    def incorporate_estimates(self, action_probs, value, up_to):
        """
        Call if the node has just been expanded via `select_leaf` to
        incorporate the prior action probabilities and state value estimated
        by the neural network.
        :param action_probs: Action probabilities for the current node's state
        predicted by the neural network.
        :param value: Value of the current node's state predicted by the neural
        network.
        :param up_to: The node to propagate until.
        """
        # A done node (i.e. episode end) should not go through this code path.
        # Rather it should directly call `backup_value` on the final node.
        # TODO: Add assert here
        # Another thread already expanded this node in the meantime.
        # Ignore wasted computation but correct visit counts.
        # if self.is_expanded:
        #     self.revert_visits(up_to=up_to)
        #     return
        #print(self.state, self.type, self.action)
        self.is_expanded = True
        self.original_prior = self.child_prior = action_probs
        if self.type == 'node':
            for action in self.n_actions:
                new_state = self.TreeEnv.next_state(self.state, action, self.type)
                idx = self.action_idx[action]
                if (str(self.state),str(new_state)) in MCTSNode.map:
                    # self.bad[idx] = 1.0 - (0.8)**(MCTSNode.map[(str(self.state),str(new_state))])
                    # self.bad[idx] = 1.0/MCTSNode.map[(str(self.state),str(new_state))]
                    self.bad[idx] = 1.0
                    #self.bad[idx] = 0.0

                if (self.state == new_state).all():
                    self.bad[idx] = 0.0
        # This is a deviation from the paper that led to better results in
        # practice (following the MiniGo implementation).
        #self.child_W = np.ones([len(self.n_actions)], dtype=np.float32) * value

        if self.type == 'board':
            for i in range(3):
                self.child_prior[i+1] += (self.child_prior[0]*(1-STOP_CUTTER))/3
            self.child_prior[0] = self.child_prior[0]*STOP_CUTTER

        self.backup_value(value, up_to=up_to)

    def backup_value(self, value, up_to):
        """
        Propagates a value estimation up to the root node.
        :param value: Value estimate to be propagated.
        :param up_to: The node to propagate until.
        """
        self.W += value
        if self.parent is None or self is up_to:
            return
        # self.parent.backup_value(value, up_to)
        # self.parent.backup_value(value*(0.9 + STOP_CUTTER/10.0), up_to)
        # self.parent.backup_value(value*1.05, up_to)
        self.parent.backup_value(value*0.997, up_to)

    def is_done(self):
        if self.is_expanded and sum(self.bad) == 0.0:
            return True
        return self.TreeEnv.is_done_state(self.depth)

    def inject_noise(self):
        dirch = np.random.dirichlet([D_NOISE_ALPHA] * len(self.n_actions))
        self.child_prior = self.child_prior * 0.75 + dirch * 0.25

        if self.type == 'board':
            for i in range(3):
                self.child_prior[i+1] += (self.child_prior[0]*(1-STOP_CUTTER))/3
            self.child_prior[0] = self.child_prior[0]*STOP_CUTTER

    def visits_as_probs(self, squash=False):
        """
        Returns the child visit counts as a probability distribution.
        :param squash: If True, exponentiate the probabilities by a temperature
        slightly large than 1 to encourage diversity in early steps.
        :return: Numpy array of shape (n_actions).
        """
        for i in range(len(self.n_actions)):
            if self.n_actions[i] in self.children:
                self.child_N[i] = self.children[self.n_actions[i]].N
        probs = self.child_N
        if squash:
            probs = probs ** .95
        return probs / np.sum(probs)

    # def print_tree(self, level=0):
    #     node_string = "\033[94m|" + "----"*level
    #     node_string += "Node: action={}\033[0m".format(self.action)
    #     node_string += "\n• state:\n{}".format(self.state)
    #     node_string += "\n• N={}".format(self.N)
    #     node_string += "\n• score:\n{}".format(self.child_action_score)
    #     node_string += "\n• Q:\n{}".format(self.child_Q)
    #     node_string += "\n• P:\n{}".format(self.child_prior)
    #     print(node_string)
    #     for _, child in sorted(self.children.items()):
    #         child.print_tree(level+1)

class MCTS:
    """
    Represents a Monte-Carlo search tree and provides methods for performing
    the tree search.
    """

    def __init__(self, board_netw, node_netw, score_netw, representation, TreeEnv, seconds_per_move=None,
                 simulations_per_move=800, num_parallel=1):
        """
        :param agent_netw: Network for predicting action probabilities and
        state value estimate.
        :param TreeEnv: Static class that defines the environment dynamics,
        e.g. which state follows from another state when performing an action.
        :param seconds_per_move: Currently unused.
        :param simulations_per_move: Number of traversals through the tree
        before performing a step.
        :param num_parallel: Number of leaf nodes to collect before evaluating
        them in conjunction.
        """
        self.board_netw = board_netw
        self.node_netw = node_netw
        self.score_netw = score_netw
        self. representation = representation
        self.TreeEnv = TreeEnv
        self.seconds_per_move = seconds_per_move
        self.simulations_per_move = simulations_per_move
        self.num_parallel = num_parallel
        self.temp_threshold = None        # Overwritten in initialize_search

        self.root = None
        MCTSNode.map.clear()
        MCTSNode.count = 0.0

        self.score_map = {}
        self.score_states = []
        self.scores = []

        self.feature_mat = self.representation.step_model.step(TreeEnv.features, self.TreeEnv.adj)

    def initialize_search(self, state=None):
        init_state = self.TreeEnv.initial_state()
        n_actions = []
        n_actions.append(-1)
        for i in range(len(init_state)):
            if init_state[i] == 1:
                n_actions.append(i)
        self.root = MCTSNode(init_state, n_actions, self.TreeEnv, type='board')
        # Number of steps into the episode after which we always select the
        # action with highest action probability rather than selecting randomly
        self.temp_threshold = TEMP_THRESHOLD

        self.searches_pi_board = []
        self.sts_board = []

        self.searches_pi_node = []
        self.sts_node = []

        for i in range(self.TreeEnv.adj.shape[0]):
            self.searches_pi_node.append([])
            self.sts_node.append([])


    def prep_input(self, state):
        #one hot
        r = []
        for i in range(len(state)):
            if state[i] > 0:
                r.append(i)
        inp = np.zeros((self.TreeEnv.adj.shape[0], self.TreeEnv.n_resources))
        for i in range(len(r)):
            inp[r[i]][i] = 1.0
        # inp = np.hstack((inp, np.zeros((inp.shape[0], 1), dtype=inp.dtype)))
        # for i in range(inp.shape[0]):
        #     inp[i][-1] = self.TreeEnv.P_val[i]
        return inp

    def tree_search(self, num_parallel=None):
        """
        Performs multiple simulations in the tree (following trajectories)
        until a given amount of leaves to expand have been encountered.
        Then it expands and evalutes these leaf nodes.
        :param num_parallel: Number of leaf states which the agent network can
        evaluate at once. Limits the number of simulations.
        :return: The leaf nodes which were expanded.
        """
        if num_parallel is None:
            num_parallel = self.num_parallel
        leaves = []
        # Failsafe for when we encounter almost only done-states which would
        # prevent the loop from ever ending.
        failsafe = 0
        while failsafe < 10:
            failsafe += 1
            leaf = self.root.select_leaf()
            # If we encounter done-state, we do not need the agent network to
            # bootstrap. We can backup the value right away.

            # if leaf.depth - self.root.depth >= 20:
            #     value = self.TreeEnv.get_return_real(leaf.state)
            #     leaf.backup_value(value, up_to=self.root)
            #     continue

            if leaf.is_done() or leaf.is_stopper:
                if mode == 'train':
                    value = self.TreeEnv.get_return_real(leaf.state)
                    if self.hash(leaf.state) not in self.score_map:
                        self.score_states.append(self.prep_input(leaf.state))
                        self.scores.append(value)
                        self.score_map[self.hash(leaf.state)] = 1
                else:
                    value = self.score_netw.step_model.step(self.prep_input(leaf.state), self.feature_mat, self.TreeEnv.adj)

                leaf.backup_value(value, up_to=self.root)
                continue
            if leaf.type == 'board':
                # input prep
                x, y = self.board_netw.step_model.step(self.prep_input(leaf.state), self.feature_mat, self.TreeEnv.adj)
                x = x.data.numpy()
                y = y.data.numpy()
            else:
                idx = leaf.parent.idx_action[leaf.action]
                x, y = self.node_netw[idx].step_model.step(self.prep_input(leaf.state),self.feature_mat, self.TreeEnv.adj)
                x = x.data.numpy()
                y = y.data.numpy()

            #print("----------> ", y)
            leaf.incorporate_estimates(x, y, up_to=self.root)
            break

    def pick_action(self):
        """
        Selects an action for the root state based on the visit counts.
        """
        for i in range(len(self.root.n_actions)):
            if self.root.n_actions[i] in self.root.children:
                self.root.child_N[i] = self.root.children[self.root.n_actions[i]].N

        # print(self.root.state, self.root.type, self.root.n_actions)
        if mode == 'test':
            if self.root.type == 'board':
                print("kahini ki ", self.root.original_prior, self.root.child_prior)
                print("pailam ki ", self.root.child_W, self.root.child_N+1, self.root.child_W/(self.root.child_N+1))

            else:
                print("magic mamoni ", self.root.n_actions, self.root.bad)

        if self.root.depth > self.temp_threshold:
            action_idx = np.argmax(self.root.bad*self.root.child_N)
            #print(self.root.bad, self.root.child_N, action_idx)
            return action_idx
        else:
            cdf = (self.root.bad*self.root.child_N).cumsum()
            cdf /= cdf[-1]
            selection = rd.random()
            action_idx = cdf.searchsorted(selection)
            assert self.root.child_N[action_idx] != 0
            return action_idx


    def hash(self, alloc):
        sum = 0.0
        poow = 1.0
        for i in range(len(alloc)):
            sum += alloc[i]*poow
            poow *= 2.0
        return sum

    def take_action(self, action):
        """
        Takes the specified action for the root state. The subsequent child
        state becomes the new root state of the tree.
        :param action: Action to take for the root state.
        """
        # Store data to be used as experience tuples.
        ##st = self.TreeEnv.get_obs_for_states([self.root.state])
        if self.root.type == 'board':
            self.sts_board.append(self.root.state)
            self.searches_pi_board.append(self.root.visits_as_probs())
            #print(self.hash(self.root.state), self.root.type, self.root.n_actions, self.root.child_N)
        else:
            idx = self.root.parent.idx_action[self.root.action]
            self.sts_node[idx].append(self.root.state)
            self.searches_pi_node[idx].append(self.root.visits_as_probs())
            #print(self.hash(self.root.state), self.root.type, idx, self.root.n_actions, self.root.child_N)


        # if self.root.type == 'board':
        #     print(action, self.root.idx_action[action])
        # if self.root.type == 'board' and self.root.idx_action[action] == -1:
        #     self.root.is_stopper = True
        #     return self

        # Resulting state becomes new root of the tree.
        new_root = self.root.maybe_add_child(action)
        #print(self.root.depth, new_root.depth)
        self.root = new_root
        #print("////////////////////////////////////////////////////////////// ", self.root.depth)
        #print("bokhri", len(MCTSNode.map))
        #del self.root.parent.children


def execute_episode(board_netw, node_netw, score_netw, representation, num_simulations, TreeEnv, curr_mode, stop_cutter):

    global STOP_CUTTER
    STOP_CUTTER = stop_cutter
    global mode
    mode = curr_mode

    mcts = MCTS(board_netw, node_netw, score_netw, representation, TreeEnv)

    mcts.initialize_search()

    # Must run this once at the start, so that noise injection actually affects
    # the first action of the episode.
    first_node = mcts.root.select_leaf()
    probs, vals = board_netw.step_model.step(mcts.prep_input(first_node.state), mcts.feature_mat, TreeEnv.adj)
    probs = probs.data.numpy()
    vals = vals.data.numpy()
    first_node.incorporate_estimates(probs, vals, first_node)

    while True:

        # if mcts.root.type == 'board':
        #     x, y = mcts.board_netw.step_model.step(mcts.prep_input(mcts.root.state), mcts.TreeEnv.adj)
        #     print("============boom========== ", mcts.root.state, y)

        mcts.root.inject_noise()
        current_simulations = mcts.root.N

        # We want `num_simulations` simulations per action not counting
        # simulations from previous actions.
        while mcts.root.N < current_simulations + num_simulations:
            mcts.tree_search()

        # mcts.root.print_tree()
        # print("_"*100)

        action = mcts.pick_action()
        mcts.take_action(action)

        if mcts.root.is_done() or mcts.root.is_stopper:
            print("chong ", sum(mcts.root.bad), mcts.root.depth, len(MCTSNode.map), MCTSNode.count)
            print("huahua~~~~ ", mcts.score_netw.step_model.step(mcts.prep_input(mcts.root.state), mcts.feature_mat, mcts.TreeEnv.adj))
            break

    # Computes the returns at each step from the list of rewards obtained at
    # each step. The return is the sum of rewards obtained *after* the step.

    if mode == 'train':
        rew = TreeEnv.get_return_real(mcts.root.state)
    else:
        rew = mcts.score_netw.step_model.step(mcts.prep_input(mcts.root.state), mcts.feature_mat, mcts.TreeEnv.adj)

    ret_board = []
    for sts in mcts.sts_board:
        # if mode == 'train':
        #     temp.append(rew - TreeEnv.get_return_real(sts))
        # else:
        #     temp.append(rew - mcts.score_netw.step_model.step(mcts.prep_input(sts), mcts.feature_mat, mcts.TreeEnv.adj))

        ret_board.append(rew)


    # ret_node = []
    # for i in range(TreeEnv.adj.shape[0]):
    #     ret_node.append([rew]*len(mcts.sts_node[i]))


    ret_node = []
    for i in range(TreeEnv.adj.shape[0]):
        temp = []
        for sts in mcts.sts_node[i]:
            # if mode == 'train':
            #     temp.append(rew - TreeEnv.get_return_real(sts))
            # else:
            #     temp.append(rew - mcts.score_netw.step_model.step(mcts.prep_input(sts), mcts.feature_mat, mcts.TreeEnv.adj))

            temp.append(rew)
        ret_node.append(temp)


    for i in range(len(mcts.sts_board)):
        mcts.sts_board[i] = mcts.prep_input(mcts.sts_board[i])

    for i in range(len(mcts.sts_node)):
        for j in range(len(mcts.sts_node[i])):
            mcts.sts_node[i][j] = mcts.prep_input(mcts.sts_node[i][j])

    return (mcts.sts_board, mcts.searches_pi_board, ret_board, mcts.sts_node, mcts.searches_pi_node, ret_node, mcts.score_states, mcts.scores)
