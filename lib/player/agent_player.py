import time

from lib.train.frame_window import FrameWindow


class AgentPlayer:
    def __init__(
            self,
            env,
            input_shape,
            model,
            image_pre_processor,
            logger,
            frame_delay,
            episodes
    ):
        self.__image_pre_processor = image_pre_processor
        self.__frame_window = FrameWindow(frame_shape=(input_shape.rows, input_shape.cols), size=input_shape.channels)
        self.__env = env
        self.__model = model
        self.__logger = logger
        self.__frame_delay = frame_delay
        self.__episodes = episodes

    def play(self, weights_path):
        self.__model.load(weights_path)
        self.__env.new_episode()
        episode = 0

        while episode < self.__episodes:
            if self.__env.is_episode_finished():
                self.__logger.info(f'Episode: {episode} - Finished - Env:{self.__env.previous_state_variables()}')
                self.__env.new_episode()
                self.__frame_window.reset()
                episode += 1
                continue
            self.__env.make_action(self.__next_action())
            time.sleep(self.__frame_delay)

    def __next_action(self):
        current_frame = self.__env.current_state().frame()
        self.__frame_window.append(self.__image_pre_processor.pre_process(current_frame))
        return self.__model.predict_action_from_frames(self.__frame_window.frames())
