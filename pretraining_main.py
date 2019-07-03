import pretraining_game_env
from large_network import LargeNetwork
import testing_game_env
import subprocess


def main():
    subprocess.call('docker kill $(docker ps -q)', shell=True)
    pretraining_game_env.create_database(database_size=50)
    LN = LargeNetwork()
    LN.init_dummy_model()
    LN.train_model_on_database(n_epochs=4)
    LN.plot_history()
    result = testing_game_env.test_network(n_games=4, LN=LN)
    print(result)

main()
