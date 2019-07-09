from pretraining.large_network import LargeNetwork
from pretraining import pretraining_game_env, pretraining_database
from agents.LN_agent import LNAgent
from testing.testing_game_env import test_network
import gdrive_auth
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default=None)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--log_id", type=str, default=None)
    parser.add_argument("--db_size", type=int, default=100)
    parser.add_argument("--n_epochs", type=int, default=4)
    parser.add_argument("--n_test_games", type=int, default=4)
    args = parser.parse_args()

    drive = gdrive_auth.get_drive()

    pretraining_database.create_database(db_size=args.db_size)
    pretraining_game_env.fill_database()

    LN = LargeNetwork(drive=drive)
    if args.model_id:
        LN.load_model(model_id=args.model_id, log_id=args.log_id)
    elif args.model_name:
        LN.init_model(args.model_name)

    LN.train_model_on_database(n_epochs=args.n_epochs)
    kwargs1 = {"a_id": 1, "n_id": 1, "LN": LN}
    kwargs2 = {"a_id": 2, "n_id": 1, "LN": LN}
    agent1 = LNAgent
    agent2 = LNAgent
    result = test_network(n_games=args.n_test_games, agent_1=agent1, agent_2=agent2, kwargs_1=kwargs1, kwargs_2=kwargs2)

    print(result)
    LN.upload_logs(test_results=result)
    LN.upload_model()
    # LN.plot_history()
