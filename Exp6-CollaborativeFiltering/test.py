input_file = open('ratings.csv', 'r')
dataset = []
user_dataset = {}
movie_dataset = {}
first = True
count = 0
# for line in input_file:
#     if first:
#         first = False
#         continue
#     row = line.split(",")
#     dataset.append(row[:-1])

#     user_dataset.setdefault(row[0], {})
#     user_dataset[row[0]].setdefault(row[1], float(row[2]))

#     movie_dataset.setdefault(row[1], {})
#     movie_dataset[row[1]].setdefault(row[0], float(row[2]))
    
#     avg_rating = 0.0
#     size = len(user_dataset['1'])
#     for movie_id, rating in user_dataset['1'].items():
#         avg_rating += rating
#     avg_rating /= size
#     print(avg_rating)
print(0/0)