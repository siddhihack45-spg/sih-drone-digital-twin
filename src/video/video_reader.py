import cv2
from pathlib import Path


class VideoReader:

    def __init__(self, video_path):
        self.video_path = Path(video_path)
        self.cap = cv2.VideoCapture(str(self.video_path))

        if not self.cap.isOpened():
            raise RuntimeError(
                f"Could not open video: {self.video_path}"
            )

    def get_fps(self):
        return self.cap.get(cv2.CAP_PROP_FPS)

    def get_frame_count(self):
        return int(
            self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

    def get_width(self):
        return int(
            self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        )

    def get_height(self):
        return int(
            self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

    def get_duration(self):
        fps = self.get_fps()

        if fps <= 0:
            return 0

        return self.get_frame_count() / fps

    def read_frame(self, frame_id):

        self.cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_id
        )

        success, frame = self.cap.read()

        if not success:
            return None

        return frame

    def release(self):
        self.cap.release()


if __name__ == "__main__":

    video = VideoReader(
        "data/video/drone.mp4"
    )

    print("========== VIDEO INFO ==========")
    print(f"Resolution : {video.get_width()} x {video.get_height()}")
    print(f"FPS        : {video.get_fps():.2f}")
    print(f"Frames     : {video.get_frame_count()}")
    print(f"Duration   : {video.get_duration():.2f} seconds")

    video.release()
    