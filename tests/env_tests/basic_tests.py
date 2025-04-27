import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from env_module.flag_frenzy_env import FlagFrenzyEnv
from env_module.SimulationInterface import ControllableEntityManouver

def sim_test():
    """
    Basic simulation interface test
    """
    env = FlagFrenzyEnv()

    try:
        while env.compute_reward() != 0:
            print("This is EXECUTED")
            env._tick()
            env._get_observations()

            # Example on how to perform an attack
            flagship = env.find_entity_by_name("Renhai")
            b1 = env.find_entity_by_name("B1")

            if flagship.IsAlive():
                if b1.CurrentManouver != ControllableEntityManouver.Combat:
                    env.execute_action([3, [b1.EntityId / env.max_entities, flagship.EntityId / env.max_entities, 0.0, 0.0]])
            elif b1.CurrentManouver == ControllableEntityManouver.NoManouver:
                env.execute_action([2, [b1.EntityId / env.max_entities]])
        print("Successfully ran simulation.")

    except Exception as e:
        print(f"There was error running the game simulation! {e}")

def gym_env_test():
    """
    Test the Gymnasium environment interface
    """
    env = FlagFrenzyEnv()

    try:
        obs, info = env.reset()

        for i in range(5):
            # env.action_space.sample() already returns a dict with action_type and params
            action = env.action_space.sample()
            observation, reward, term, trunc, info = env.step(action)

        env.close()
        print("Successfully stepped through gymnasium environment!")

    except Exception as e:
        print(f"There was an error running the game simulation! {e}")

if __name__ == "__main__":
    print("Running simulation test...")
    sim_test()
    print("Simulation test completed.")

    print("Running gymnasium test...")
    gym_env_test()
    print("Gymnasium test completed.")