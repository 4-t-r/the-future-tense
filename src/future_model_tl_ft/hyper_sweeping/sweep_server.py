import wandb
import math

sweep_config = {
    "name": "20ng_distilbert",
    "method": "bayes",
    "metric": {
        "name": "val_loss",
        "goal": "minimize"
    },
    "parameters": {
        "dense_layers": {
            "values": [[128,64],[128],[128,64,32],[512],[512,256],[64],[]]
        },
        "input_dropout_rate": {
            "min": 0.0, "max": 0.5,
        },
        "dense_dropout_rate": {
            "min": 0.0, "max": 0.75,
        },
        "learning_rate": {
            "min": math.log(0.0000001), "max": math.log(0.1), "distribution": "log_uniform"
            # according to https://github.com/wandb/client/issues/507
        },
    }
}


if __name__ == "__main__":
        sweep_id = wandb.sweep(sweep_config)
        print(sweep_id)