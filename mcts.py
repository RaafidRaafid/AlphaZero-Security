
import math
import random as rd
import collections
import numpy as np

# Exploration constant
c_PUCT = 1.38
# Dirichlet noise alpha parameter.
D_NOISE_ALPHA = 0.03
# Number of steps into the episode after which we always select the
# action with highest action probability rather than selecting randomly
TEMP_THRESHOLD = 5


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
        self.n_vlosses = 0  # Number of virtual losses on this node
        self.child_N = np.zeros([len(n_actions)], dtype=np.float32)
        self.child_W = np.zeros([len(n_actions)], dtype=np.float32)
        # Save copy of original prior before it gets mutated by dirichlet noise
        self.original_prior = np.zeros([len(n_actions)], dtype=np.float32)
        self.child_prior = np.zeros([len(n_actions)], dtype=np.float32)
        #print(self.child_prior)
        self.children = {}

        self.action_idx = {}
        for i in range(len(n_actions)):
            self.action_idx[n_actions[i]] = i
        self.idx_action = {}
        for i in range(len(n_actions)):
            self.idx_action[i] = n_actions[i]

        #print(self.type, self.state, self.n_actions, self.action, self)
        #print(self)

    @property
    def N(self):
        """
        Returns the current visit count of the node.
        """
        return self.parent.child_N[self.action]

    @N.setter
    def N(self, value):
        self.parent.child_N[self.action] = value

    @property
    def W(self):
        """
        Returns the current total value of the node.
        """
        return self.parent.child_W[self.action]

    @W.setter
    def W(self, value):
        self.parent.child_W[self.action] = value

    @property
    def Q(self):
        """
        Returns the current action value of the node.
        """
        return self.W / (1 + self.N)

    @property
    def child_Q(self):
        return self.child_W / (1 + self.child_N)

    @property
    def child_U(self):
        #print("-", self.type, self.child_prior, self.child_N, self.state, self)
        return (c_PUCT * math.sqrt(1 + self.N) *
                self.child_prior / (1 + self.child_N))

    @property
    def child_action_score(self):
        """
        Action_Score(s, a) = Q(s, a) + U(s, a) as in paper. A high value
        means the node should be traversed.
        """
        return self.child_Q + self.child_U

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
            # Choose action with highest score.
            best_move = np.argmax(current.child_action_score)
            current = current.maybe_add_child(best_move)
        return current

    def maybe_add_child(self, action):
        """
        Adds a child node for the given action if it does not yet exists, and
        returns it.
        :param action: Action to take in current state which leads to desired
        child node.
        :return: Child MCTSNode.
        """
        action = self.idx_action[action]
        if action not in self.children:
            # Obtain state following given action.
            #print(self.type, self.state, self.n_actions, action, self)
            new_state = self.TreeEnv.next_state(self.state, action, self.type)
            #print(self.type, self.state, self.n_actions, action, self)
            if self.type == 'board':
                self.children[action] = MCTSNode(new_state, self.TreeEnv.actions[action],
                                                self.TreeEnv, type = 'node',
                                                action=self.action_idx[action], parent=self)
            else:
                temp = [0.0]*len(new_state)
                temp[:] = new_state[:]
                ttemp = []
                for i in range(len(temp)):
                    if temp[i] > 0:
                        ttemp.append(i)
                #print(temp, action, action[0], self.state, ttemp)
                self.children[action] = MCTSNode(new_state, ttemp,
                                                self.TreeEnv, type = 'board',
                                                action=self.action_idx[action], parent=self)
            #print(">>>", self.children[action].type, self.children[action].state, self.children[action])
        return self.children[action]

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
        if self.is_expanded:
            self.revert_visits(up_to=up_to)
            return
        self.is_expanded = True
        self.original_prior = self.child_prior = action_probs
        #print(action_probs)
        # This is a deviation from the paper that led to better results in
        # practice (following the MiniGo implementation).
        self.child_W = np.ones([len(self.n_actions)], dtype=np.float32) * value
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
        self.parent.backup_value(value, up_to)

    def is_done(self):
        return self.TreeEnv.is_done_state(self.depth)

    def inject_noise(self):
        #print(self.child_prior)
        dirch = np.random.dirichlet([D_NOISE_ALPHA] * len(self.n_actions))
        self.child_prior = self.child_prior * 0.75 + dirch * 0.25

    def visits_as_probs(self, squash=False):
        """
        Returns the child visit counts as a probability distribution.
        :param squash: If True, exponentiate the probabilities by a temperature
        slightly large than 1 to encourage diversity in early steps.
        :return: Numpy array of shape (n_actions).
        """
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

    def __init__(self, board_netw, node_netw, TreeEnv, seconds_per_move=None,
                 simulations_per_move=800, num_parallel=8):
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
        self.TreeEnv = TreeEnv
        self.seconds_per_move = seconds_per_move
        self.simulations_per_move = simulations_per_move
        self.num_parallel = num_parallel
        self.temp_threshold = None        # Overwritten in initialize_search

        self.qs = []
        self.rewards = []
        self.searches_pi = []
        self.sts = []

        self.root = None

    def initialize_search(self, state=None):
        init_state = self.TreeEnv.initial_state()
        n_actions = []
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
        inp = np.hstack((inp, np.zeros((inp.shape[0], 1), dtype=inp.dtype)))
        for i in range(inp.shape[0]):
            inp[i][-1] = self.TreeEnv.P_val[i]
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
        while len(leaves) < num_parallel and failsafe < num_parallel * 2:
            failsafe += 1
            # self.root.print_tree()
            # print("_"*50)
            leaf = self.root.select_leaf()
            # If we encounter done-state, we do not need the agent network to
            # bootstrap. We can backup the value right away.
            if leaf.is_done():
                value = self.TreeEnv.get_return(leaf.state)
                leaf.backup_value(value, up_to=self.root)
                continue
            # Otherwise, discourage other threads to take the same trajectory
            # via virtual loss and enqueue the leaf for evaluation by agent
            # network.
            leaf.add_virtual_loss(up_to=self.root)
            leaves.append(leaf)
        # Evaluate the leaf-states all at once and backup the value estimates.
        if leaves:
            action_probs = []
            values = []
            for leaf in leaves:
                #print(leaf.state)
                if leaf.type == 'board':
                    # input prep
                    x, y = self.board_netw.step_model.step(self.prep_input(leaf.state), self.TreeEnv.adj)
                    x = x.data.numpy()
                    y = y.data.numpy()
                else:
                    idx = leaf.parent.idx_action[leaf.action]
                    x, y = self.node_netw[idx].step_model.step(self.prep_input(leaf.state), self.TreeEnv.adj)
                    x = x.data.numpy()
                    y = y.data.numpy()
                action_probs.append(x)
                values.append(y)
            for leaf, action_prob, value in zip(leaves, action_probs, values):
                leaf.revert_virtual_loss(up_to=self.root)
                leaf.incorporate_estimates(action_prob, value, up_to=self.root)
        return leaves

    def pick_action(self):
        """
        Selects an action for the root state based on the visit counts.
        """
        if self.root.depth > self.temp_threshold:
            action_idx = np.argmax(self.root.child_N)
        else:
            cdf = self.root.child_N.cumsum()
            cdf /= cdf[-1]
            selection = rd.random()
            action_idx = cdf.searchsorted(selection)
            assert self.root.child_N[action_idx] != 0
        return action_idx

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
        else:
            idx = self.root.parent.idx_action[self.root.action]
            self.sts_node[idx].append(self.root.state)
            self.searches_pi_node[idx].append(self.root.visits_as_probs())

        # Resulting state becomes new root of the tree.
        self.root = self.root.maybe_add_child(action)
        del self.root.parent.children


def execute_episode(board_netw, node_netw, num_simulations, TreeEnv):

    mcts = MCTS(board_netw, node_netw, TreeEnv)

    mcts.initialize_search()

    # Must run this once at the start, so that noise injection actually affects
    # the first action of the episode.
    first_node = mcts.root.select_leaf()
    probs, vals = board_netw.step_model.step(mcts.prep_input(first_node.state), TreeEnv.adj)
    probs = probs.data.numpy()
    vals = vals.data.numpy()
    first_node.incorporate_estimates(probs, vals, first_node)

    while True:
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

        if mcts.root.is_done():
            break

    # Computes the returns at each step from the list of rewards obtained at
    # each step. The return is the sum of rewards obtained *after* the step.

    #state, search_pi, q, reward
    rew = TreeEnv.get_return(mcts.root.state)
    ret_board = [rew]*len(mcts.sts_board)
    ret_node = []
    for i in range(TreeEnv.adj.shape[0]):
        ret_node.append([rew]*len(mcts.sts_node[i]))

    return (mcts.sts_board, mcts.searches_pi_board, ret_board, mcts.sts_node, mcts.searches_pi_node, ret_node)
