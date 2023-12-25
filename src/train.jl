include("extract_data.jl")
include("models.jl")

using BSON
using Flux: params
using Flux

function train_model!(m, L, X, y;
        opt = Descent(0.1),
        batchsize = 128,
        n_epochs = 50,
        file_name = "")

    println("Initiating splitting into minibatches")
    batches = Flux.Data.DataLoader((X, y); batchsize, shuffle = true)

    println("Minibatching done, starting to train")
    for i in 1:n_epochs
        println(i)
        Flux.train!(L, params(m), batches, opt)
        if mod(i, 51) == 0
            BSON.bson(file_name * "_" * string(i, base  = 10, pad = 2) * ".bson", m=m)
            println("Model saved after" * string(i, base  = 10, pad = 2) * "iterations")
        end
    end

    println("Training successfull, saving current model")
    BSON.bson(file_name, m=m)
    println("Successfully saved into " * file_name)

    return
end

function load_model!(file_name, m; force=false, kwargs...)
    m_weights = BSON.load(file_name)[:m]
    Flux.loadparams!(m, params(m_weights))
end

function train_or_load!(file_name, m, args...; force=false, kwargs...)

    !isdir(dirname(file_name)) && mkpath(dirname(file_name))

    if force || !isfile(file_name)
        train_model!(m, args...; file_name=file_name, kwargs...)
    else
        load_model!(file_name, m; force=false, kwargs...)
    end
end