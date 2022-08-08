import argparse

import wandb

from training import run_training

default_hyperparameters = {"use_wandb": False,
                           "dense_layers": [128,64],
                           "input_dropout_rate": 0.0,
                           "dense_dropout_rate": 0.2,
                           "learning_rate": 0.001
                           }


def sweep_agent():

        wandb.init(config=default_hyperparameters)

        hyperparameters = dict(wandb.config)
        # if running inside a sweep agent, this contains the default_hyperparameters with the adjustments
        # made by the sweep controller.

        hyperparameters.update({"use_wandb": True})

        run_training(**hyperparameters)


if __name__ == "__main__":
        parser = argparse.ArgumentParser(description='Run a single wandb parameter sweep agent.')
        parser.add_argument('--sweep_id')
        args = parser.parse_args()

        wandb.agent(sweep_id=args.sweep_id,
                                function=sweep_agent)