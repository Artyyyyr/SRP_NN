import SRP_NN

import os.path

import cv2
import time
import uuid

import numpy as np
import mediapipe as mp
from mediapipe.python.solutions.drawing_utils import DrawingSpec

import tensorflow as tf
import matplotlib.pyplot as plt

labels = {1: "Scissors", 2: "Rock", 3: "Paper", 4: "Start", 5: "Stats"}


def stats(game):
    ratio_my_win = []
    ratio_nn_win = []
    ratio_draw = []
    my_win = 0
    nn_win = 0
    draw = 0

    for i in range(len(game.story)):
        if game.story[i] == 0:
            draw += 1
        elif game.story[i] == 1:
            nn_win += 1
        elif game.story[i] == 2:
            my_win += 1

        ratio_nn_win.append(nn_win/(nn_win + my_win + draw))
        ratio_my_win.append(my_win/(nn_win + my_win + draw))
        ratio_draw.append(draw/(nn_win + my_win + draw))

    print("My wins: " + str(my_win))
    print("NN wins: " + str(nn_win))
    print("Draws: " + str(draw))

    plt.figure("Stats")
    plt.plot(ratio_nn_win, label="NN wins")
    plt.plot(ratio_my_win, label="My wins")
    plt.plot(ratio_draw, label="Draws")

    plt.legend()
    plt.xlabel("Number of games")
    plt.ylabel("Percentages")

    plt.show()


def mp_detection(model, img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img.flags.writeable = False
    res = model.process(img)
    img.flags.writeable = True
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    return img, res


mp_holistic = mp.solutions.holistic
mp_draw = mp.solutions.drawing_utils


def draw_landmarks(img, res):
    mp_draw.draw_landmarks(img, res.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                           DrawingSpec(color=(255, 148, 255), thickness=2, circle_radius=3))
    mp_draw.draw_landmarks(img, res.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                           DrawingSpec(color=(148, 255, 148), thickness=2, circle_radius=3))
    mp_draw.draw_landmarks(img, res.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                           DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=0))


def get_landmarks_array(res):
    right_hand_lm = np.zeros(63)
    if res.right_hand_landmarks:
        for i in range(len(res.right_hand_landmarks.landmark)):
            right_hand_lm[3 * i] = res.right_hand_landmarks.landmark[i].x
            right_hand_lm[3 * i + 1] = res.right_hand_landmarks.landmark[i].y
            right_hand_lm[3 * i + 2] = res.right_hand_landmarks.landmark[i].z

    return np.concatenate([right_hand_lm])


def start(model, hidden_size, srp_load_path=True):
    game = SRP_NN.Game(hidden_size=hidden_size, load_path=srp_load_path)
    sequence = []
    threshold = 0.8
    cap = cv2.VideoCapture(0)
    rock_frames, scissors_frames, paper_frames, start_frames, nothing_frames, stats_frames = 0, 0, 0, 0, 0, 0
    start_threshold = 20
    stats_threshold = 30
    forget_threshold = 60
    forget_threshold_gesture = 20
    gesture_threshold = 20
    phase = 'start'

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            s, img = cap.read()

            img, res = mp_detection(holistic, img)
            draw_landmarks(img, res)

            points = get_landmarks_array(res)
            sequence.append(points)
            sequence = sequence[-30:]

            if len(sequence) == 30:
                prediction = model.predict(np.expand_dims(sequence, axis=0), verbose=0)
                if prediction[0][0] > threshold:
                    # cv2.putText(img, 'Rock', (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
                    rock_frames += 1
                    scissors_frames = 0
                    paper_frames = 0
                    start_frames = 0
                    nothing_frames = 0
                    stats_frames = 0
                elif prediction[0][1] > threshold:
                    # cv2.putText(img, 'Scissors', (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
                    rock_frames = 0
                    scissors_frames += 1
                    paper_frames = 0
                    start_frames = 0
                    nothing_frames = 0
                    stats_frames = 0
                elif prediction[0][2] > threshold:
                    # cv2.putText(img, 'Paper', (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
                    rock_frames = 0
                    scissors_frames = 0
                    paper_frames += 1
                    start_frames = 0
                    nothing_frames = 0
                    stats_frames = 0
                elif prediction[0][3] > threshold:
                    # cv2.putText(img, 'Start', (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
                    rock_frames = 0
                    scissors_frames = 0
                    paper_frames = 0
                    start_frames += 1
                    nothing_frames = 0
                    stats_frames = 0
                elif prediction[0][4] > threshold:
                    rock_frames = 0
                    scissors_frames = 0
                    paper_frames = 0
                    start_frames = 0
                    nothing_frames = 0
                    stats_frames += 1
                else:
                    rock_frames = 0
                    scissors_frames = 0
                    paper_frames = 0
                    start_frames = 0
                    nothing_frames += 1

            get_landmarks_array(res)

            if nothing_frames > 10000:
                nothing_frames = 0

            if phase == 'start':
                if start_frames > start_threshold:
                    phase = 'choice'
                elif stats_frames > stats_threshold:
                    stats(game)
                    stats_frames = 0
            elif phase == 'choice' and nothing_frames > forget_threshold:
                phase = 'start'
            elif phase == 'choice' and rock_frames > gesture_threshold:
                phase = 'rock'
                nn_choice = labels[game.update(2)]
            elif phase == 'choice' and scissors_frames > gesture_threshold:
                phase = 'scissors'
                nn_choice = labels[game.update(1)]
            elif phase == 'choice' and paper_frames > gesture_threshold:
                phase = 'paper'
                nn_choice = labels[game.update(3)]
            elif phase == 'rock' and scissors_frames + paper_frames + start_frames + nothing_frames > forget_threshold_gesture:
                phase = 'start'
            elif phase == 'scissors' and rock_frames + paper_frames + start_frames + nothing_frames > forget_threshold_gesture:
                phase = 'start'
            elif phase == 'paper' and rock_frames + scissors_frames + start_frames + nothing_frames > forget_threshold_gesture:
                phase = 'start'

            if phase == 'rock':
                if game.story[-1] == 0:
                    cv2.putText(img, nn_choice, (300, 250),
                                cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 2, cv2.LINE_AA)
                elif game.story[-1] == 1:
                    cv2.putText(img, nn_choice, (300, 250),
                                cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 2, cv2.LINE_AA)
                elif game.story[-1] == 2:
                    cv2.putText(img, nn_choice, (300, 250),
                                cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2, cv2.LINE_AA)
            elif phase == 'scissors':
                if game.story[-1] == 0:
                    cv2.putText(img, nn_choice, (300, 250),
                                cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 2, cv2.LINE_AA)
                elif game.story[-1] == 1:
                    cv2.putText(img, nn_choice, (300, 250),
                                cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 2, cv2.LINE_AA)
                elif game.story[-1] == 2:
                    cv2.putText(img, nn_choice, (300, 250),
                                cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2, cv2.LINE_AA)
            elif phase == 'paper':
                if game.story[-1] == 0:
                    cv2.putText(img, nn_choice, (300, 250),
                                cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 2, cv2.LINE_AA)
                elif game.story[-1] == 1:
                    cv2.putText(img, nn_choice, (300, 250),
                                cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 2, cv2.LINE_AA)
                elif game.story[-1] == 2:
                    cv2.putText(img, nn_choice, (300, 250),
                                cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.putText(img, phase, (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
            if 'prediction' in locals():
                cv2.putText(img, "Rock: " + str(round(prediction[0][0] * 1000)/10) + "%", (0, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(img, "Scissors: " + str(round(prediction[0][1] * 1000)/10) + "%", (0, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(img, "Paper: " + str(round(prediction[0][2] * 1000)/10) + "%", (0, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(img, "Start: " + str(round(prediction[0][3] * 1000) / 10) + "%", (0, 250),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(img, "Stats: " + str(round(prediction[0][4] * 1000) / 10) + "%", (0, 300),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow("Camera", img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


def write_data():
    cap = cv2.VideoCapture(0)
    num_frames = 30

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            s, img = cap.read()

            if cv2.waitKey(1) & 0xFF == ord("r"):

                cv2.putText(img, 'Rock', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(img, 'Wait 3 second)))', (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

                cv2.imshow("Camera", img)
                cv2.waitKey(1)

                time.sleep(3)

                path = "images/Rock/Rock_{}".format(uuid.uuid1())
                os.makedirs(path)
                print("Rock saved in \"" + path + "\"")

                for i in range(num_frames):
                    s, img = cap.read()

                    img, res = mp_detection(holistic, img)
                    draw_landmarks(img, res)

                    cv2.imwrite(path + "/Rock_img" + str(i) + ".jpg", img)

                    array = get_landmarks_array(res)
                    np.save(path + "/Rock_numpy" + str(i) + ".npy", array)

                    cv2.imshow("Camera", img)
                    cv2.waitKey(1)

            if cv2.waitKey(1) & 0xFF == ord("s"):
                cv2.putText(img, 'Scissors', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(img, 'Wait 3 second)))', (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2,
                            cv2.LINE_AA)

                cv2.imshow("Camera", img)
                cv2.waitKey(1)

                time.sleep(3)

                path = "images/Scissors/Scissors_{}".format(uuid.uuid1())
                os.makedirs(path)
                print("Scissors saved in \"" + path + "\"")

                for i in range(num_frames):
                    s, img = cap.read()

                    img, res = mp_detection(holistic, img)
                    draw_landmarks(img, res)

                    cv2.imwrite(path + "/Scissors_img" + str(i) + ".jpg", img)

                    array = get_landmarks_array(res)
                    np.save(path + "/Scissors_numpy" + str(i) + ".npy", array)

                    cv2.imshow("Camera", img)
                    cv2.waitKey(1)

            if cv2.waitKey(1) & 0xFF == ord("p"):
                cv2.putText(img, 'Paper', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(img, 'Wait 3 second)))', (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2,
                            cv2.LINE_AA)

                cv2.imshow("Camera", img)
                cv2.waitKey(1)

                time.sleep(3)

                path = "images/Paper/Paper_{}".format(uuid.uuid1())
                os.makedirs(path)
                print("Paper saved in \"" + path + "\"")

                for i in range(num_frames):
                    s, img = cap.read()

                    img, res = mp_detection(holistic, img)
                    draw_landmarks(img, res)

                    cv2.imwrite(path + "/Paper_img" + str(i) + ".jpg", img)

                    array = get_landmarks_array(res)
                    np.save(path + "/Paper_numpy" + str(i) + ".npy", array)

                    cv2.imshow("Camera", img)
                    cv2.waitKey(1)

            if cv2.waitKey(1) & 0xFF == ord("t"):
                cv2.putText(img, 'Start', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(img, 'Wait 3 second)))', (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2,
                            cv2.LINE_AA)

                cv2.imshow("Camera", img)
                cv2.waitKey(1)

                time.sleep(3)

                path = "images/Start/Start_{}".format(uuid.uuid1())
                os.makedirs(path)
                print("Start saved in \"" + path + "\"")

                for i in range(num_frames):
                    s, img = cap.read()

                    img, res = mp_detection(holistic, img)
                    draw_landmarks(img, res)

                    cv2.imwrite(path + "/Start_img" + str(i) + ".jpg", img)

                    array = get_landmarks_array(res)
                    np.save(path + "/Start_numpy" + str(i) + ".npy", array)

                    cv2.imshow("Camera", img)
                    cv2.waitKey(1)

            if cv2.waitKey(1) & 0xFF == ord("n"):
                cv2.putText(img, 'Nothing', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(img, 'Wait 3 second)))', (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2,
                            cv2.LINE_AA)

                cv2.imshow("Camera", img)
                cv2.waitKey(1)

                time.sleep(3)

                path = "images/Nothing/Nothing_{}".format(uuid.uuid1())
                os.makedirs(path)
                print("Start saved in \"" + path + "\"")

                for i in range(num_frames):
                    s, img = cap.read()

                    img, res = mp_detection(holistic, img)
                    draw_landmarks(img, res)

                    cv2.imwrite(path + "/Nothing_img" + str(i) + ".jpg", img)

                    array = get_landmarks_array(res)
                    np.save(path + "/Nothing_numpy" + str(i) + ".npy", array)

                    cv2.imshow("Camera", img)
                    cv2.waitKey(1)

            if cv2.waitKey(1) & 0xFF == ord("a"):
                cv2.putText(img, 'Stats', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(img, 'Wait 3 second)))', (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2,
                            cv2.LINE_AA)

                cv2.imshow("Camera", img)
                cv2.waitKey(1)

                time.sleep(3)

                path = "images/Stats/Stats_{}".format(uuid.uuid1())
                os.makedirs(path)
                print("Start saved in \"" + path + "\"")

                for i in range(num_frames):
                    s, img = cap.read()

                    img, res = mp_detection(holistic, img)
                    draw_landmarks(img, res)

                    cv2.imwrite(path + "/Stats_img" + str(i) + ".jpg", img)

                    array = get_landmarks_array(res)
                    np.save(path + "/Stats_numpy" + str(i) + ".npy", array)

                    cv2.imshow("Camera", img)
                    cv2.waitKey(1)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            cv2.imshow("Camera", img)

    cap.release()
    cv2.destroyAllWindows()


def load_train():
    sequences = []
    labels = []
    num_frames = 30

    for folder in os.listdir("images/Rock"):
        sequence = []
        for i in range(num_frames):
            sequence.append(np.load("images/Rock/" + folder + "/Rock_numpy" + str(i) + ".npy"))

        sequences.append(np.array(sequence))
        labels.append(0)

    for folder in os.listdir("images/Scissors"):
        sequence = []
        for i in range(num_frames):
            sequence.append(np.load("images/Scissors/" + folder + "/Scissors_numpy" + str(i) + ".npy"))

        sequences.append(np.array(sequence))
        labels.append(1)

    for folder in os.listdir("images/Paper"):
        sequence = []
        for i in range(num_frames):
            sequence.append(np.load("images/Paper/" + folder + "/Paper_numpy" + str(i) + ".npy"))

        sequences.append(np.array(sequence))
        labels.append(2)

    for folder in os.listdir("images/Start"):
        sequence = []
        for i in range(num_frames):
            sequence.append(np.load("images/Start/" + folder + "/Start_numpy" + str(i) + ".npy"))

        sequences.append(np.array(sequence))
        labels.append(3)

    for folder in os.listdir("images/Stats"):
        sequence = []
        for i in range(num_frames):
            sequence.append(np.load("images/Stats/" + folder + "/Stats_numpy" + str(i) + ".npy"))

        sequences.append(np.array(sequence))
        labels.append(4)

    labels = tf.keras.utils.to_categorical(labels).tolist()

    sequence = []
    for i in range(num_frames):
        sequence.append(np.zeros(63))

    sequences.append(np.array(sequence))
    labels.append([0.2, 0.2, 0.2, 0.2, 0.2])

    for folder in os.listdir("images/Nothing"):
        sequence = []
        for i in range(num_frames):
            sequence.append(np.load("images/Nothing/" + folder + "/Nothing_numpy" + str(i) + ".npy"))

        sequences.append(np.array(sequence))
        labels.append([0.2, 0.2, 0.2, 0.2, 0.2])

    return np.array(sequences), np.array(labels)


# 0 - Rock, 1 - Scissors, 2 - Paper, 3 - Start, 4 - Stats
def train():
    x, y = load_train()

    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.LSTM(30, return_sequences=False, activation='relu', input_shape=(30, 63)))

    model.add(tf.keras.layers.Dense(200, activation='relu'))
    model.add(tf.keras.layers.Dense(100, activation='relu'))
    model.add(tf.keras.layers.Dense(5, activation='softmax'))

    callback = tf.keras.callbacks.TensorBoard(log_dir="Log/")
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])
    model.fit(x, y, epochs=4000, callbacks=[callback])

    return model
