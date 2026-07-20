import jax
import jax.numpy as jnp


def make_static_penalty_function(alpha):

    def penalty_function(state, control):
        return alpha * jnp.mean(state.est.epistemic_std)

    return penalty_function


def make_difference_penalty_function(alpha):

    def penalty_function(state, control):
        return alpha * (jnp.mean(state.est.epistemic_std) - state.last_unc)
    
    return penalty_function


def make_pos_difference_penalty_function(alpha):

    def penalty_function(state, control):
        return alpha * jnp.maximum(jnp.mean(state.est.epistemic_std) - state.last_unc, 0.0)
    
    return penalty_function

def make_pos_difference_reward_function(alpha):

    def reward_function(state, control):
        return alpha * jnp.maximum(state.last_unc - jnp.mean(state.est.epistemic_std), 0.0)
    
    return reward_function


def make_ratio_penalty_function(alpha):

    def penalty_function(state, control):
        return alpha * jnp.mean(state.est.epistemic_std) / (state.last_unc + 1e-8)
    
    return penalty_function


def make_mixed_penalty_function(*args):

    def penalty_function(state, control):
        pen = 0
        for penalty in args:
            pen += penalty(state, control)
        return pen
    
    return penalty_function