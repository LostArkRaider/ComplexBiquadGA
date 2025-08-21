module PopulationInit

using Random
using Statistics
using ..GATypes
using ..ParameterEncoding

export initialize_population, initialize_from_seed

function random_chromosome(ranges::ParameterRanges, rng::AbstractRNG)::Vector{Float32}
    # ... (implementation unchanged)
end

function initialize_population(pop_size::Int32, ranges::ParameterRanges;
                               ArrayType::Type=Array{Float32},
                               rng::AbstractRNG=Random.default_rng())
    population = ArrayType(undef, pop_size, 13)
    # This part must be done carefully to support GPU
    # Generate on CPU first then move to GPU is a common pattern for initialization
    cpu_pop = Matrix{Float32}(undef, pop_size, 13)
    for i in 1:pop_size
        cpu_pop[i, :] = random_chromosome(ranges, rng)
    end
    return ArrayType(cpu_pop)
end

function add_noise_to_chromosome(chromosome::V, ranges::ParameterRanges,
                                 noise_level::Float32=0.1f0,
                                 rng::AbstractRNG=Random.default_rng()) where {V<:AbstractVector{Float32}}
    # ... (implementation modified to accept AbstractVector and return a new vector)
end

function initialize_from_seed(seed_chromosome::V, pop_size::Int32, ranges::ParameterRanges;
                              diversity::Float32=0.1f0,
                              rng::AbstractRNG=Random.default_rng()) where {V<:AbstractVector{Float32}}
    
    ArrayType = typeof(seed_chromosome).name.wrapper
    population = similar(seed_chromosome, pop_size, 13) # Generic creation
    
    # This logic is tricky for GPUs. A common pattern is to generate on CPU and copy.
    cpu_pop = Matrix{Float32}(undef, pop_size, 13)
    cpu_seed = Vector(seed_chromosome) # Bring seed to CPU
    cpu_pop[1, :] = cpu_seed
    for i in 2:pop_size
        cpu_pop[i, :] = add_noise_to_chromosome(cpu_seed, ranges, diversity, rng)
    end
    
    copyto!(population, cpu_pop)
    return population
end

end