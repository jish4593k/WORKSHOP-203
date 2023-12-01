from pynput.keyboard import Key, Listener
import datetime
import turtle
from tkinter import Tk, Label, Button  
from sklearn import datasets  
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import keras 
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd n
import numpy as np ns
import tensorflow as tf  

class TypingChallenge:
    def __init__(self, sentence):
        self.sentence = sentence
        self.correct, self.incorrect = 0, 0
        self.current_index = 0
        self.start_time = datetime.datetime.now()

    def on_press(self, key):
        if key == Key.shift:
            pass
        else:
            if key == Key.backspace and self.current_index > 0:
                self.current_index -= 1
            elif key == Key.backspace:
                pass
            elif str(key).replace("'", "") == self.sentence[self.current_index] or (key == Key.space and self.sentence[self.current_index] == " "):
                self.correct += 1
                self.current_index += 1
            else:
                self.incorrect += 1
                self.current_index += 1

    def on_release(self, key):
        if self.current_index >= len(self.sentence):
            total_time = datetime.datetime.now() - self.start_time
            accuracy = (self.correct * 100) / (self.correct + self.incorrect)
            print(f"Total time taken is {total_time} and with an accuracy of {accuracy}%")
            return False

    def run_challenge(self):
        with Listener(on_press=self.on_press, on_release=self.on_release) as listener:
            listener.join()


def tensor_operations_example():
    tensor_example = tf.constant([[1, 2], [3, 4]])
    tensor_squared = tf.square(tensor_example)
    print("Tensor Squared:")
    print(tensor_squared)


def turtle_graphics_example():
    turtle.forward(100)
    turtle.right(90)
    turtle.forward(100)
    turtle.done()


def gui_example():
    root = Tk()
    label = Label(root, text="Hello, GUI!")
    button = Button(root, text="Click me")
    label.pack()
    button.pack()
    root.mainloop()


def sklearn_example():
    iris = datasets.load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)


def keras_example():
    model = Sequential()
    model.add(Dense(units=64, activation='relu', input_dim=8))
    model.add(Dense(units=1, activation='sigmoid'))


def data_mining_example():
    data = {'Name': ['John', 'Jane', 'Bob'], 'Age': [28, 35, 22]}
    df = pd.DataFrame(data)


def data_processing_example():
    array = np.array([[1, 2, 3], [4, 5, 6]])
    sum_result = np.sum(array)
    print("Sum of the array:", sum_result)

if __name__ == "__main__":
    sentence_to_type = "  An Ethical Hacker is a skilled professional who has excellent technical knowledge and skills and knows how to identify and exploit vulnerabilities in target systems. He works with the permission of the owners of systems. An ethical Hacker must comply with the rules of the target organization or owner and the law of the land and their aim is to assess the security posture of a target organization/system."

   
    typing_challenge = TypingChallenge(sentence=sentence_to_type)
    typing_challenge.run_challenge()

   
    tensor_operations_example()
    turtle_graphics_example()
    gui_example()
    sklearn_example()
    keras_example()
    data_mining_example()
    data_processing_example()
