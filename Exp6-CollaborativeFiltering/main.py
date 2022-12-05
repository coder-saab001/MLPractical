import sys
import math
import os


class Recommender_System:
    def __init__(self, input_file_name, user_id, movie_id, k):
        self.input_file_name = input_file_name
        self.user_id = user_id
        self.movie_id = movie_id
        self.k = k
        self.dataset = None
        self.user_dataset = None
        self.movie_dataset = None
        if not os.path.isfile(self.input_file_name):
            self.quit("Invalid file")
        self.dataset, self.user_dataset, self.movie_dataset = self.load_data(self.input_file_name)
        if self.user_id not in self.user_dataset.keys():
            self.quit("Invalid user ID")

    def pearson_correlation(self, user1, user2):
        result = 0.0
        user1_data = self.user_dataset[user1]
        user2_data = self.user_dataset[user2]

        rx_avg = self.user_average_rating(user1_data)
        ry_avg = self.user_average_rating(user2_data)
        sxy = self.common_items(user1_data, user2_data)

        top_result = 0.0
        bottom_left_result = 0.0
        bottom_right_result = 0.0
        for item in sxy:
            rxs = user1_data[item]
            rys = user2_data[item]
            top_result += (rxs - rx_avg) * (rys - ry_avg)
            bottom_left_result += pow((rxs - rx_avg), 2)
            bottom_right_result += pow((rys - ry_avg), 2)
        bottom_left_result = math.sqrt(bottom_left_result)
        bottom_right_result = math.sqrt(bottom_right_result)
        if bottom_right_result == 0:
            return 1000
        result = top_result / (bottom_left_result * bottom_right_result)

        return result

    def user_average_rating(self, user_data):
        avg_rating = 0.0
        size = len(user_data)
        for (movie_id, rating) in user_data.items():
            avg_rating += float(rating)
        avg_rating /= size * 1.0
        return avg_rating

    def common_items(self, user1_data, user2_data):
        result = []
        ht = {}
        for (movie_id, rating) in user1_data.items():
            ht.setdefault(movie_id, 0)
            ht[movie_id] += 1
        for (movie_id, rating) in user2_data.items():
            ht.setdefault(movie_id, 0)
            ht[movie_id] += 1
        for (k, v) in ht.items():
            if v == 2:
                result.append(k)
        return result

    def k_nearest_neighbors(self, user, k):
        neighbors = []
        result = []
        for (user_id, data) in self.user_dataset.items():
            if user_id == user:
                continue
            upc = self.pearson_correlation(user, user_id)
            if upc != 1000:
                neighbors.append([user_id, upc])
        sorted_neighbors = sorted(neighbors, key=lambda neighbors: (neighbors[1], neighbors[0]), reverse=True)

        for i in range(k):
            if i >= len(sorted_neighbors):
                break
            result.append(sorted_neighbors[i])
        return result

    def predict(self, user, item, k_nearest_neighbors):
        valid_neighbors = self.check_neighbors_validattion(item, k_nearest_neighbors)
        if not len(valid_neighbors):
            return 0.0
        top_result = 0.0
        bottom_result = 0.0
        for neighbor in valid_neighbors:
            neighbor_id = neighbor[0]
            neighbor_similarity = neighbor[1]
            rating = self.user_dataset[neighbor_id][item]
            top_result += neighbor_similarity * rating
            bottom_result += neighbor_similarity
        result = top_result / bottom_result
        self.display(valid_neighbors, result)
        return result

    def check_neighbors_validattion(self, item, k_nearest_neighbors):
        result = []
        for neighbor in k_nearest_neighbors:
            neighbor_id = neighbor[0]
            if item in self.user_dataset[neighbor_id].keys():
                result.append(neighbor)
        return result

    def load_data(self, input_file_name):
        input_file = open(input_file_name, 'r')
        dataset = []
        user_dataset = {}
        movie_dataset = {}
        first = True
        for line in input_file:
            if first:
                first = False
                continue
            row = str(line)
            row = row.split(",")
            dataset.append(row[:-1])

            user_dataset.setdefault(row[0], {})
            user_dataset[row[0]].setdefault(row[1], float(row[2]))

            movie_dataset.setdefault(row[1], {})
            movie_dataset[row[1]].setdefault(row[0], float(row[2]))
        return dataset, user_dataset, movie_dataset

    def display(self, k_nearest_neighbors, prediction):
        for neighbor in k_nearest_neighbors:
            print(neighbor[0], neighbor[1])
        print(prediction)


if __name__ == '__main__':
    input_file_name = sys.argv[1]
    user_id = sys.argv[2]
    movie_id = sys.argv[3]
    k = int(sys.argv[4])
    rs = Recommender_System(input_file_name, user_id, movie_id, k)
    k_nearest_neighbors = rs.k_nearest_neighbors(user_id, k)
    prediction = rs.predict(user_id, movie_id, k_nearest_neighbors)