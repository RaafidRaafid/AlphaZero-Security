
class staticEnv:

    @staticmethod
    def next_state(state, action):
        raise NotImplementedError

    @staticmethod
    def is_done_state(state, step_idx):
        raise NotImplementedError

    @staticmethod
    def initial_state():
        raise NotImplementedError

    @staticmethod
    def get_obs_for_states(states):
        raise NotImplementedError

    '''
    ajaira?
    '''

    @staticmethod
    def get_return(state, step_idx):
        raise NotImplementedError

    @staticmethod
    def get_return_real(state, step_idx):
        raise NotImplementedError
