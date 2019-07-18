import gym

import pickle
import os
import numpy as np
import random
import time
import traceback
import math

import tensorflow as tf

from keras.optimizers import Adam
from keras.models import Model
from keras.layers import *
from keras import backend as K

import rl.core

import multiprocessing as mp

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw


# 各クラスは別ファイル想定です。(全コードでは1ファイルにまとめています)
from PendulumProcessorForDQN import PendulumProcessorForDQN
from R2D2Manager import R2D2Manager
from ObservationLogger import ObservationLogger
from MovieLogger import MovieLogger

# global
agent = None
logger = None


ENV_NAME = "Pendulum-v0"
def create_processor():
    return PendulumProcessorForDQN(enable_image=False)

def create_processor_image():
    return PendulumProcessorForDQN(enable_image=True, image_size=84)

def create_optimizer():
    return Adam(lr=0.00025)

def actor_func(index, actor, callbacks):
    env = gym.make(ENV_NAME)
    if index == 0:
        verbose = 1
    else:
        verbose = 0
    actor.fit(env, nb_steps=200_000, visualize=False, verbose=verbose, callbacks=callbacks)

#--------------------------------------

def main(image):
    global agent, logger
    env = gym.make(ENV_NAME)

    if image:
        processor = create_processor_image
        input_shape = (84, 84)
    else:
        processor = create_processor
        input_shape = env.observation_space.shape

    # 引数
    args = {
        # model関係
        "input_shape": input_shape, 
        "enable_image_layer": image, 
        "nb_actions": 5, 
        "input_sequence": 4,     # 入力フレーム数
        "dense_units_num": 64,  # Dense層のユニット数
        "metrics": [],           # optimizer用
        "enable_dueling_network": True,  # dueling_network有効フラグ
        "dueling_network_type": "ave",   # dueling_networkのアルゴリズム
        "enable_noisynet": False,        # NoisyNet有効フラグ
        "lstm_type": "lstm",             # LSTMのアルゴリズム
        "lstm_units_num": 64,   # LSTM層のユニット数

        # learner 関係
        "remote_memory_capacity": 100_000,    # 確保するメモリーサイズ
        "remote_memory_warmup_size": 200,    # 初期のメモリー確保用step数(学習しない)
        "remote_memory_type": "per_proportional", # メモリの種類
        "per_alpha": 0.8,        # PERの確率反映率
        "per_beta_initial": 0.0,     # IS反映率の初期値
        "per_beta_steps": 100_000,   # IS反映率の上昇step数
        "per_enable_is": False,      # ISを有効にするかどうか
        "batch_size": 16,            # batch_size
        "target_model_update": 1500, #  target networkのupdate間隔
        "enable_double_dqn": True,   # DDQN有効フラグ
        "enable_rescaling_priority": False,   # rescalingを有効にするか(priotrity)
        "enable_rescaling_train": False,      # rescalingを有効にするか(train)
        "rescaling_epsilon": 0.001,  # rescalingの定数
        "burnin_length": 20,        # burn-in期間
        "priority_exponent": 0.9,    # priority優先度

        # actor 関係
        "local_memory_update_size": 50,    # LocalMemoryからRemoteMemoryへ投げるサイズ
        "actor_model_sync_interval": 500,  # learner から model を同期する間隔
        "gamma": 0.99,      # Q学習の割引率
        "epsilon": 0.3,        # ϵ-greedy法
        "epsilon_alpha": 1,    # ϵ-greedy法
        "multireward_steps": 1, # multistep reward
        "action_interval": 1,   # アクションを実行する間隔

        # その他
        "load_weights_path": "",  # 保存ファイル名
        #"load_weights_path": "qiita08r2d2.h5",  # 読み込みファイル名
        "save_weights_path": "qiita08_r2d2_image_lstm.h5",  # 保存ファイル名
        "save_overwrite": True,   # 上書き保存するか
        "logger_interval": 10,    # ログ取得間隔(秒)
        "enable_GPU": False,      # GPUを使うか
    }


    manager = R2D2Manager(
        actor_func=actor_func, 
        num_actors=1,    # actor数
        args=args, 
        create_processor_func=processor,
        create_optimizer_func=create_optimizer,
    )

    agent, learner_logs, actors_logs = manager.train()

    #--- plot
    plot_logs(learner_logs, actors_logs)

    #--- test
    agent.processor.mode = "test"  # env本来の報酬を返す
    agent.test(env, nb_episodes=5, visualize=True)

    view = MovieLogger()   # 動画用
    logger = ObservationLogger()
    agent.test(env, nb_episodes=1, visualize=False, callbacks=[view, logger])
    view.view(interval=1, gifname="anim1.gif")  # 動画用

    #--- NNの可視化
    if image:
        plt.figure(figsize=(8.0, 6.0), dpi = 100)  # 大きさを指定
        plt.axis('off')
        #ani = matplotlib.animation.FuncAnimation(plt.gcf(), plot, frames=150, interval=5)
        ani = matplotlib.animation.FuncAnimation(plt.gcf(), plot, frames=len(logger.observations), interval=5)

        #ani.save('anim2.mp4', writer="ffmpeg")
        ani.save('anim2.gif', writer="imagemagick", fps=60)
        #plt.show()


if __name__ == '__main__':
    # コメントアウトで切り替え
    main(image=False)
    #main(image=True)
