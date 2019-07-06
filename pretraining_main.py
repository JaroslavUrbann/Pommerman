import pretraining_game_env
import pretraining_database
from large_network import LargeNetwork
import testing_game_env
import subprocess
import gdrive_auth


def main():
    # subprocess.call('docker kill $(docker ps -q)', shell=True)
    drive = gdrive_auth.get_drive()
    pretraining_database.create_database(db_size=50)
    LN = LargeNetwork()
    LN.load_model(drive, id="1HPKpEAf0p9AOu7ePXGU19ivxCbMx4YKH")

    # LN.init_dummy_model()
    LN.train_model_on_database(n_epochs=4)
    # LN.plot_history()
    # result = testing_game_env.test_network(n_games=4, LN=LN)
    result = [3, 2, 5]
    LN.upload_logs(drive, id="1u2YVOUtA6kxPWHm-SL5_zBRprOCNWKqT", name=None, test_results=result)
    LN.upload_model(drive, name=None)
    # print(result)

main()
