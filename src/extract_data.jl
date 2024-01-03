# Loads and prepares data to feed into NN
using DataFrames
using CSV

max_color = 255
T = Float32
image_size = 96
image_half = 48

function show_image(X, y, idx)
    # println(size(img))
    img = reshape(X[:, :, :, idx], image_size, image_size);
    keypoints = (y[:, idx] .* 48) .+ 48
    keypoints = reshape(keypoints, 2, 15)

    plot(Gray.(reshape(img, image_size, image_size))')
    plot!(keypoints[1,:], keypoints[2,:], seriestype=:scatter)
end

# Loads training data and removes any row with missing colums
function load_trn_data(path::String)
    trn_data = DataFrame(CSV.File(path))
    dropmissing!(trn_data)
    keypoints = reshape_keypoints(trn_data)
    select!(trn_data, :Image)
    images = reshape_images(trn_data)
    return reshape(images, image_size, image_size, 1, size(trn_data.Image, 1)), keypoints
end

# Loads public dataset
function load_tst_data(path::String)
    tst_data = DataFrame(CSV.File(path))
    images = reshape_images(tst_data)
    return reshape(images, image_size, image_size, 1, size(tst_data.Image, 1))
end

# Reshapes images into matrices and scales intensity from [0, 255] to [0, 1]
function reshape_images(df::DataFrame; T=T)
    reshaped = parse.(T, reduce(vcat, split.(df[!, :Image]," ")))
    return (reshaped ./ max_color)'
end

# Reshapes keypoints so that they range from -1 to 1
function reshape_keypoints(df::DataFrame; T=T)
    y = Matrix{T}(df[:, Not(:Image)])
    y = (y .- image_half) ./ image_half
    return y'
end