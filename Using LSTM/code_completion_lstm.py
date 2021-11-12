import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
import numpy


class Code_Completion_Lstm:
    def token_to_string(self, token):
        return token["type"] + "-@@-" + token["value"]

    def string_to_token(self, string):
        splitted = string.split("-@@-")
        return {"type": splitted[0], "value": splitted[1]}

    # Returns an hot vector with all entries 0.
    # For padding purpose this is required
    def zero_hot(self):
        vector = [0] * len(self.string_to_number)
        return vector

    def one_hot(self, string):
        vector = [0] * len(self.string_to_number)
        vector[self.string_to_number[string]] = 1
        return vector

    def prepare_data(self, token_lists):
        # encode tokens into one-hot vectors
        all_token_strings = set()
        for token_list in token_lists:
            for token in token_list:
                all_token_strings.add(self.token_to_string(token))
        all_token_strings = list(all_token_strings)
        all_token_strings.sort()
        print("Unique tokens: " + str(len(all_token_strings)))
        self.string_to_number = dict()
        self.number_to_string = dict()
        max_number = 0
        for token_string in all_token_strings:
            self.string_to_number[token_string] = max_number
            self.number_to_string[max_number] = token_string
            max_number += 1

        xs = []
        ys = []
        for token_list in token_lists:
            length = len(token_list)
            for idx, token in enumerate(token_list):


                # At the start padding all the zero vectors and as output a
                # e.g. [0,0,0] -> [a]
                if idx == 0:
                    token_string = self.token_to_string(token)
                    previous_3_token_string = self.zero_hot()
                    previous_2_token_string = self.zero_hot()
                    previous_1_token_string = self.zero_hot()

                # If index is 1 then consider previous token and padd left zeros
                # e.g. [0,0,a] -> [b]
                if idx == 1:
                    token_string = self.token_to_string(token)
                    previous_3_token_string = self.zero_hot()
                    previous_2_token_string = self.zero_hot()
                    previous_1_token_string = self.one_hot(self.token_to_string(token_list[idx - 1]))

                # If index is 2 then consider previous tokens and padd left zeros
                # e.g. [0,a,b] -> [c]
                if idx == 2:
                    token_string = self.token_to_string(token)
                    previous_3_token_string = self.zero_hot()
                    previous_2_token_string = self.one_hot(self.token_to_string(token_list[idx - 2]))
                    previous_1_token_string = self.one_hot(self.token_to_string(token_list[idx - 1]))

                # e.g. [a,b,c] -> [d]
                if idx > 2:
                    token_string = self.token_to_string(token)
                    previous_3_token_string = self.one_hot(self.token_to_string(token_list[idx - 3]))
                    previous_2_token_string = self.one_hot(self.token_to_string(token_list[idx - 2]))
                    previous_1_token_string = self.one_hot(self.token_to_string(token_list[idx - 1]))

                if(length - idx) > 4:
                    next_1_token_string = self.one_hot(self.token_to_string(token_list[idx+1]))
                    next_2_token_string = self.one_hot(self.token_to_string(token_list[idx+2]))

                if(length - idx) == 4:
                    next_1_token_string = self.one_hot(self.token_to_string(token_list[idx+1]))
                    next_2_token_string = self.one_hot(self.token_to_string(token_list[idx+2]))


                if(length - idx) == 3:
                    next_1_token_string = self.one_hot(self.token_to_string(token_list[idx+1]))
                    next_2_token_string = self.one_hot(self.token_to_string(token_list[idx+2]))


                if(length - idx) == 2:
                    next_1_token_string = self.one_hot(self.token_to_string(token_list[idx+1]))
                    next_2_token_string = self.zero_hot()


                if(length - idx) == 1:
                    next_1_token_string = self.zero_hot()
                    next_2_token_string = self.zero_hot()


                #previous_1_token_string is the immediate previous token string to current token_string.
                # previous_3_token_string is the farthest previous token string to current token_string.

                xs.append([next_2_token_string, next_1_token_string, previous_3_token_string, previous_2_token_string,
                           previous_1_token_string])
                ys.append(self.one_hot(token_string))

        #This part is done for 2 holes
                if idx == 0:
                    token_string = self.token_to_string(token)
                    #previous_4_token_string = self.zero_hot()
                    previous_3_token_string = self.zero_hot()
                    previous_2_token_string = self.zero_hot()
                    previous_1_token_string = self.zero_hot()


                if idx == 1:
                    token_string = self.token_to_string(token)
                    previous_3_token_string = self.zero_hot()
                    previous_2_token_string = self.zero_hot()
                    previous_1_token_string = self.one_hot(self.token_to_string(token_list[idx - 1]))

                if idx == 2:
                    token_string = self.token_to_string(token)
                    previous_3_token_string = self.zero_hot()
                    previous_2_token_string = self.one_hot(self.token_to_string(token_list[idx - 2]))
                    previous_1_token_string = self.one_hot(self.token_to_string(token_list[idx - 1]))


                if idx > 2:
                    token_string = self.token_to_string(token)
                    previous_3_token_string = self.one_hot(self.token_to_string(token_list[idx - 3]))
                    previous_2_token_string = self.one_hot(self.token_to_string(token_list[idx - 2]))
                    previous_1_token_string = self.one_hot(self.token_to_string(token_list[idx - 1]))

                if(length - idx) > 5:
                    next_1_token_string = self.one_hot(self.token_to_string(token_list[idx+2]))
                    next_2_token_string = self.one_hot(self.token_to_string(token_list[idx+3]))

                if(length - idx) == 5:
                    next_1_token_string = self.one_hot(self.token_to_string(token_list[idx+2]))
                    next_2_token_string = self.one_hot(self.token_to_string(token_list[idx+3]))
                if(length - idx) == 4:
                    next_1_token_string = self.one_hot(self.token_to_string(token_list[idx+2]))
                    next_2_token_string = self.one_hot(self.token_to_string(token_list[idx+3]))

                if(length - idx) == 3:
                    next_1_token_string = self.one_hot(self.token_to_string(token_list[idx+2]))
                    next_2_token_string = self.zero_hot()

                if(length - idx) <= 2:
                    next_1_token_string = self.zero_hot()
                    next_2_token_string = self.zero_hot()

                xs.append([next_2_token_string, next_1_token_string,  previous_3_token_string, previous_2_token_string,
                           previous_1_token_string])
                ys.append(self.one_hot(token_string))


        print("x,y pairs: " + str(len(xs)))
        return (xs, ys)

    def create_network(self):
        self.net = tflearn.input_data(shape=[None, 5, len(self.string_to_number)])
        self.net = tflearn.lstm(self.net, 200, return_seq=True)
        self.net = tflearn.lstm(self.net, 200)
        self.net = tflearn.fully_connected(self.net, len(self.string_to_number), activation='sigmoid', bias=True,
                                           trainable=True)
        self.net = tflearn.regression(self.net, optimizer='RMSprop', loss='categorical_crossentropy')
        self.model = tflearn.DNN(self.net)

    def load(self, token_lists, model_file):
        self.prepare_data(token_lists)
        self.create_network()
        self.model.load(model_file)

    def train(self, token_lists, model_file):
        (xs, ys) = self.prepare_data(token_lists)
        self.create_network()
        self.model.fit(xs, ys, n_epoch=1, batch_size=50, show_metric=True)
        self.model.save(model_file)

    def predict(self, x):
        y = self.model.predict([x])
        predicted_seq = y[0]
        if type(predicted_seq) is numpy.ndarray:
            predicted_seq = predicted_seq.tolist()
        best_number = predicted_seq.index(max(predicted_seq))
        best_string = self.number_to_string[best_number]
        best_token = self.string_to_token(best_string)
        return best_token

    def query(self, prefix, suffix):

        # Padding all input vectors with zero as there is no prefix available
        if len(prefix) == 0:

            previous_3_token_string = self.zero_hot()
            previous_2_token_string = self.zero_hot()
            previous_1_token_string = self.zero_hot()
        elif len(prefix) == 1:
            # Use the available prefix tokens and padd the rest with zero vectors
            previous_3_token_string = self.zero_hot()
            previous_2_token_string = self.zero_hot()
            previous_1_token_string = self.one_hot(self.token_to_string(prefix[-1]))
        elif len(prefix) == 2:
            # Use the available prefix tokens and padd the rest with zero vectors
            #previous_4_token_string = self.zero_hot()
            previous_3_token_string = self.zero_hot()
            previous_2_token_string = self.one_hot(self.token_to_string(prefix[-2]))
            previous_1_token_string = self.one_hot(self.token_to_string(prefix[-1]))

        else:
            # This is the case for most of the time
            # Use all the available prefix tokens as an input

            previous_3_token_string = self.one_hot(self.token_to_string(prefix[-3]))
            previous_2_token_string = self.one_hot(self.token_to_string(prefix[-2]))
            previous_1_token_string = self.one_hot(self.token_to_string(prefix[-1]))


        # padding suffix
        if len(suffix) == 0:
            next_1_token_string = self.zero_hot()
            next_2_token_string = self.zero_hot()

        elif len(suffix) == 1:
            next_1_token_string = self.one_hot(self.token_to_string(suffix[0]))
            next_2_token_string = self.zero_hot()


        elif len(suffix) == 2:
            next_1_token_string = self.one_hot(self.token_to_string(suffix[0]))
            next_2_token_string = self.one_hot(self.token_to_string(suffix[1]))


        else:
            next_1_token_string = self.one_hot(self.token_to_string(suffix[0]))
            next_2_token_string = self.one_hot(self.token_to_string(suffix[1]))

        predicted = []
        x = [next_2_token_string, next_1_token_string, previous_3_token_string, previous_2_token_string, previous_1_token_string]
        best_token = self.predict(x)
        predicted.append(best_token)

        iterations = 0
        while True:


            previous_3_token_string = previous_2_token_string
            previous_2_token_string = previous_1_token_string
            previous_1_token_string = self.one_hot(self.token_to_string(best_token))

            x = [next_2_token_string, next_1_token_string,  previous_3_token_string, previous_2_token_string, previous_1_token_string]
            best_token = self.predict(x)

			# Prediction would keep on going until the predicted token matches the suffix first token OR a boundary which is set to 10 is reached.
            if (len(suffix) != 0 and (best_token['value'] == suffix[0]['value'] and best_token['type'] == suffix[0][
                'type']) or iterations == 6):
                # If the prediction goes on till 6 iterations, then it is very unlikely that prediction would be correct.
                # So for that return only the first predicted token.
                if (iterations >= 5):
                    return [predicted[0]]
                else:
					#return all the predicted tokens
                    return predicted
            else:
                predicted.append(best_token)

            iterations = iterations + 1

########################################################
