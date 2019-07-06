import pretraining_game_env
import pretraining_database
from large_network import LargeNetwork
import testing_game_env
import subprocess
import gdrive_auth


def main():
    # subprocess.call('docker kill $(docker ps -q)', shell=True)
    pretraining_database.create_database(db_size=50)
    LN = LargeNetwork()
    LN.init_dummy_model()
    LN.train_model_on_database(n_epochs=4)
    # LN.plot_history()
    # result = testing_game_env.test_network(n_games=4, LN=LN)
    result = [3, 2, 1]
    drive = gdrive_auth.get_drive()
    LN.upload_logs(drive, name="asdf", test_results=result)
    LN.upload_model(drive, name="asdf")
    # print(result)

main()
