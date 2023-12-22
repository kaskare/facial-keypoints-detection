include("extract_data.jl")

model_cnn = Chain(
    Conv((3, 3), 1=>16, relu),
    MaxPool((2,2)),
    Conv((3, 3), 16=>32, relu),
    MaxPool((2,2)),
    flatten,
    Dense(288, 30),
    softmax
) |> gpu

# out1 = model(noisy |> gpu) |> cpu 