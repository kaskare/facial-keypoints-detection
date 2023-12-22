# Loads and prepares data to feed into NN
using DataFrames
using CSV

max_color = 255
T = Float32

function load_trn_data(path::String)
    trn_data = DataFrame(CSV.File(path))
    dropmissing!(trn_data)
    keypoints = reshape_keypoints(select(trn_data,  Not(:Image)))
    images = reshape_images(trn_data)
    return images, keypoints
end

function load_tst_data(path::String)
    tst_data = DataFrame(CSV.File(path))
    dropmissing!(tst_data)
    images = reshape_images(tst_data)
    return reshape(images, 96, 96, 1, 1783)
end

# Reshapes images into matrices and scales intensity from [0, 255] to [0, 1]
function reshape_images(df::DataFrame; T=T)
    reshaped = parse.(T, mapreduce(permutedims, vcat, split.(df.Image," ")))
    return reshaped / max_color
end

# Reshapes keypoints to s scale of -1 to 1
function reshape_keypoints(df::DataFrame; T=T)
    y = Matrix{T}(df)
    y = (y .- (length(y)/2)) / length(y)
    return y
end 

# trn_images, keypoints = load_trn_data("/data/training.csv")
# tst_images = load_tst_data("/data/test.csv")
# println(images[1, :])
