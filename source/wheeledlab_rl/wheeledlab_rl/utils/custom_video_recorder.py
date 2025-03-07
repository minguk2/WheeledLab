import os
from typing import Callable

import gymnasium as gym
import wandb
from gymnasium import error, logger
from gymnasium.wrappers.rendering import RecordVideo


class CustomRecordVideo(RecordVideo):
    def __init__(
        self,
        env: gym.Env,
        video_folder: str,
        episode_trigger: Callable[[int], bool] = None,
        step_trigger: Callable[[int], bool] = None,
        video_length: int = 0,
        name_prefix: str = "rl-video",
        disable_logger: bool = False,
        enable_wandb: bool = True,
    ):
        if enable_wandb and wandb.run.name is None:
            raise ValueError("wandb must be initialized before wrapping.")

        super().__init__(
            env,
            video_folder,
            episode_trigger,
            step_trigger,
            video_length,
            name_prefix,
            disable_logger,
        )
        self.enable_wandb = enable_wandb

    def stop_recording(self):
        """Stop current recording and saves the video."""
        assert self.recording, "stop_recording was called, but no recording was started"

        if len(self.recorded_frames) == 0:
            logger.warn("Ignored saving a video as there were zero frames to save.")
        else:
            try:
                from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
            except ImportError as e:
                raise error.DependencyNotInstalled(
                    'MoviePy is not installed, run `pip install "gymnasium[other]"`'
                ) from e

            clip = ImageSequenceClip(self.recorded_frames, fps=60)
            moviepy_logger = None if self.disable_logger else "bar"
            path = os.path.join(self.video_folder, f"{self._video_name}.mp4")
            clip.write_videofile(path, logger=moviepy_logger)
            if self.enable_wandb:
                wandb.log({"Video": wandb.Video(path)}, commit=False)

        self.recorded_frames = []
        self.recording = False
        self._video_name = None
