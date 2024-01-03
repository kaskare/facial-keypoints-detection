include("extract_data.jl")
include("models.jl")
include("train.jl")

using Flux
using BSON
using Plots
using Colors

# Model architectures are in models.jl to choose from
m = model2
ps = Flux.params(m)

# Load the data
X_train, y_train = load_trn_data(TRAINING_CSV_PATH)
X_test = load_tst_data(TESTING_CSV_PATH)
println("Data loaded")

# # Cross entropy
# L(x,y) = Flux.crossentropy(m(X_train), y_train)

# RMSE - root mean squared distance
L(x,y) = sqrt(Flux.mse(m(X_train), y_train))

grad = gradient(() -> L(X_train, y_train), ps)

train_model!(m, L, X_train, y_train; file_name=MODEL_PATH)

y_test = m(X_test)
show_image(X_test, y_test, 1)
