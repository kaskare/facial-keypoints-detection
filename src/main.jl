include("extract_data.jl")
include("models.jl")
include("train.jl")

using Flux
using BSON
using Plots
using Colors

# Model architectures are in models.jl to choose from
m = model2

function show_image(X, y, idx)
    
    img = reshape(X[:, :, :, idx], image_size, image_size);
    # println(size(img))
    
    keypoints = (y[:, idx] .* 48) .+ 48
    keypoints = reshape(keypoints, 2, 15)
    # println(size(keypoints))

    plot(Gray.(reshape(img, image_size, image_size))')
    plot!(keypoints[1,:], keypoints[2,:], seriestype=:scatter)
end

# Load the data
X_train, y_train = load_trn_data(TRAINING_CSV_PATH)
X_test = load_tst_data(TESTING_CSV_PATH)
println("Data loaded")

ps = Flux.params(m)

# # Cross entropy
# L(x,y) = Flux.crossentropy(m(X_train), y_train)

# RMSE
L(x,y) = sqrt(Flux.mse(m(X_train), y_train))

grad = gradient(() -> L(X_train, y_train), ps)

train_or_load!(MODEL_PATH, m, L, X_train, y_train)
# load_model!(MODEL_PATH, m)

y_test = m(X_test)
show_image(X_test, y_test, 1)
