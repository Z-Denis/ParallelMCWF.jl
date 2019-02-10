"""
Module providing parallelised versions of [`QuantumOptics.timeevolution.mcwf`](@ref).
"""
module ParallelMCWF

using Distributed, Base.Threads
using ProgressMeter, JLD2
import OrdinaryDiffEq
using QuantumOptics.bases, QuantumOptics.states, QuantumOptics.operators
using QuantumOptics.operators_dense, QuantumOptics.operators_sparse
using QuantumOptics.timeevolution
using QuantumOptics.operators_lazysum, QuantumOptics.operators_lazytensor, QuantumOptics.operators_lazyproduct
@everywhere using QuantumOptics.timeevolution.timeevolution_mcwf
const DecayRates = Union{Vector{Float64}, Matrix{Float64}, Nothing}
Base.@pure pure_inference(fout,T) = Core.Compiler.return_type(fout, T)

include("trajs_IO.jl")
export load_trajs, save_trajs
include("trajs_processing.jl")
export kets_to_dm, kets_to_obs
include("pmcwf.jl")
export pmcwf

end # module
