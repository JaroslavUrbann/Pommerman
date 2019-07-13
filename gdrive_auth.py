from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials
import time
from threading import Thread


gauth = None


def refresh_token():
    global gauth
    while True:
        time.sleep(10 * 60)
        if gauth.access_token_expired:
            gauth.Refresh()


# google colab only
def get_drive():
    global gauth
    auth.authenticate_user()
    gauth = GoogleAuth()
    gauth.credentials = GoogleCredentials.get_application_default()
    refresh = Thread(target=refresh_token)
    refresh.start()
    return GoogleDrive(gauth)