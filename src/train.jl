include("extract_data.jl")
include("models.jl")

using BSON
using Flux: params
using Flux

function train_model!(m, L, X, y;
        opt = Descent(0.1),
        batchsize = 128,
        n_epochs = 50,
        file_name = "/home/askar/Desktop/facial-keypoints-detection/models/unnamed",
        checkpoint_interval = 100,
        checkpoint = 0)

    batches = Flux.Data.DataLoader((X, y); batchsize, shuffle = true)
    println("Minibatching done, starting to train")

    for i in (checkpoint + 1):n_epochs
        Flux.train!(L, params(m), batches, opt)
        if mod(i, checkpoint_interval) == 0
            BSON.bson(file_name * "_$(i).bson", m=m)
            println("Model saved after $(i) iterations")
        end
    end

    println("Training successfull, saving current model")
    BSON.bson(file_name, m=m)
    println("Successfully saved into " * file_name)
    return
end

function load_model!(file_name, m; force=false, kwargs...)
    m_weights = BSON.load(file_name * ".bson")[:m]
    Flux.loadparams!(m, params(m_weights))
end

function train_from_chackpoint!(m, L, X, y, checkpoint;
        opt = Descent(0.1),
        batchsize = 128,
        n_epochs = 50,
        file_name = "/home/askar/Desktop/facial-keypoints-detection/models/unnamed")

    load_model(file_name * "_$(checkpoint)", m)
    train_model!(m, L, X, y;
                opt = opt,
                batchsize = batchsize,
                n_epochs = n_epochs,
                file_name = file_name,
                checkpoint = checkpoint)
end