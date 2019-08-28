from RL.game_env import train_network
from pretraining.network import Network
from RL.chat_network import ChatNetwork
import gdrive_auth
import numpy as np

model_id = "1-N2tLjFSgxS2ZFLhd1kQFe60wwMZ2_di"
model_name = "small_N1_RL"

chat_model_id = ""
chat_model_name = "N1_chat"

drive = None
drive = gdrive_auth.get_drive()


def do_network_stuff(N, CN):
    train_network(model=N.model, chat_model=CN.model, n_steps=20)


def get_models():
    N = Network(drive=drive, name=model_name, model_id=model_id)
    N.load_model()
    # N.init_model()
    CN = ChatNetwork(drive=drive, name=chat_model_name, model_id=chat_model_id)

    w = N.model.layers[1].get_weights()
    w[0][:, :, 21:25, 0:256] = np.random.uniform(-0.01, 0.01, size=(3, 3, 4, 256))
    N.model.layers[1].set_weights(w)

    l = 40
    while l < len(N.model.layers):
        w = N.model.layers[l - 1].get_weights()
        N.model.layers[l].set_weights(w)
        l += 2

    if chat_model_id:
        CN.load_model()
    elif chat_model_name:
        CN.init_model()

    return N, CN


N, CN = get_models()
do_network_stuff(N, CN)