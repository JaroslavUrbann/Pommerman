import cv2
import numpy as np
from PIL import Image
import os
import base64
from IPython.display import HTML


# Class that creates and displays a video
class VideoMaker:
    _sprite_folder = "/content/Pommerman/CustomSprites"
    _mp4_path = "vid.mp4"
    _video_path = "video.mp4"
    _mappings = {
        "231, 76, 60": "GreenBomberman.png",
        "46, 139, 87": "RedBomberman.png",
        "65, 105, 225": "YellowBomberman.png",
        "238, 130, 238": "BlueBomberman.png",
        "240, 248, 255": "BackgroundTile.png",
        "128, 128, 128": "SolidBlock.png",
        "210, 180, 140": "ExplodableBlock.png",
        "255, 153, 51": "BombBackground.png",
        "241, 196, 15": "FlameBackground.png",
        "141, 137, 124": "fog",  # fog
        "153, 153, 255": "BombPowerupBackground.png",
        "153, 204, 204": "FlamePowerupBackground.png",
        "97, 169, 169": "SpeedPowerupBackground.png",
        "48, 117, 117": "agent"  # dummy agent
    }

    def __init__(self):
        os.system("rm " + self._mp4_path)
        os.system("rm " + self._video_path)
        self._coded_frames = []
        self.fps = 5

    def add_coded_frame(self, frame):
        self._coded_frames.append(frame)

    def _decode_square(self, code):
        key = ', '.join([str(int(n)) for n in code])
        path = os.path.join(self._sprite_folder, self._mappings[key])
        img = Image.open(path).convert('RGB')
        return np.array(img)

    def _decode_frame(self, frame):
        hd_frame = np.zeros((64 * 11, 64 * 11, 3), dtype="uint8")
        for y in range(11):
            for x in range(11):
                square = self._decode_square(frame[y, x])
                hd_frame[y * 64:(y + 1) * 64, x * 64:(x + 1) * 64] = square
        return hd_frame

    def _create_video(self):
        out = cv2.VideoWriter(self._mp4_path, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (64 * 11, 64 * 11))
        for i in range(len(self._coded_frames)):
            next_frame = self._decode_frame(self._coded_frames[i])
            out.write(next_frame)
        #             print("next frame " + str(i))
        out.release()

    def show_video(self):
        self._create_video()
        os.system("ffmpeg -i " + self._mp4_path + " -vcodec libx264 " + self._video_path + " -y")
        video = open(self._video_path, "rb").read()
        encoded = base64.b64encode(video)
        return HTML(data='''<video alt="test" autoplay 
                 loop controls style="height: 704px;width: 704px;">
                 <source src="data:video/mp4;base64,{0}" type="video/mp4" />
              </video>'''.format(encoded.decode('ascii')))