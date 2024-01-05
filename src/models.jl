using Flux

TRAINING_CSV_PATH = "/home/askar/Desktop/facial-keypoints-detection/data/training_100_img.csv"
TESTING_CSV_PATH = "/home/askar/Desktop/facial-keypoints-detection/data/test_1_img.csv"
MODEL_PATH = "/home/askar/Desktop/facial-keypoints-detection/models/model2"

# model 1
model = Chain(
    Conv((3, 3), 1=>32, relu),
    MaxPool((2,2)),
    Conv((3, 3), 32=>64, relu),
    MaxPool((2,2)),
    Conv((3, 3), 64=>128, relu),
    MaxPool((2,2)),
    x -> reshape(x, :, size(x, 4)),
    Dense(12800, 4608, relu),
    Dense(4608, 512, relu),
    Dense(512, 30)
)

# model 2
model2 = Chain(
    Conv((3, 3), 1=>8, relu),
    MaxPool((2,2)),
    x -> reshape(x, :, size(x, 4)),
    Dense(17672, 7744),
    Dense(7744, 3522),
    Dense(3522, 30)
)

# old
model_cnn = Chain(
    Conv((3, 3), 1=>8, relu),
    MaxPool((2, 2)),
    Conv((3, 3), 8=>4, relu),
    MaxPool((2,2)),
    x -> reshape(x, :, size(x, 4)),
    Dense(1936, 1000, relu),
    Dense(1000, 500, relu),
    Dense(500, 100, relu),
    Dense(100, 30),
)

