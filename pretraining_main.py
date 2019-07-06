import pretraining_game_env
import pretraining_database
from large_network import LargeNetwork
import testing_game_env
import subprocess
import gdrive_auth
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default=None)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--log_id", type=str, default=None)
    parser.add_argument("--db_size", type=int, default=50)
    parser.add_argument("--n_epochs", type=int, default=4)
    parser.add_argument("--n_test_games", type=int, default=4)
    args = parser.parse_args()

    subprocess.call('docker kill $(docker ps -q)', shell=True)
    drive = gdrive_auth.get_drive()

    pretraining_database.create_database(db_size=args.db_size)
    pretraining_game_env.fill_database()

    LN = LargeNetwork(drive)
    if args.model_id and args.log_id:
        LN.load_model(model_id=args.model_id, log_id=args.log_id)
    elif args.model_name:
        LN.init_model(args.model_name)

    LN.train_model_on_database(n_epochs=args.n_epochs)

    result = testing_game_env.test_network(n_games=args.n_test_games, LN=LN)
    print(result)
    LN.upload_logs(test_results=result)
    LN.upload_model()
    LN.plot_history()
