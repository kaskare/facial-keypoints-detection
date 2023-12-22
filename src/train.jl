include("extract_data.jl")

using Flux
using CUDA

# Add your path to data here
TRAINING_CSV_PATH = "/home/askar/Desktop/facial-keypoints-detection/data/training.csv"
TESTING_CSV_PATH = "/home/askar/Desktop/facial-keypoints-detection/data/test.csv"

# X_train, y_train = load_trn_data("/home/askar/Desktop/facial-keypoints-detection/data/training.csv")
X_test = load_tst_data(TRAINING_CSV_PATH)

# Define the CNN architecture
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
    softmax
)

# out1 = model(noisy |> gpu) |> cpu 
model_cnn(X_test)