using ParallelMCWF
using Test
using QuantumOptics

@testset "ParallelMCWF.jl" begin
    # Write your own tests here.
    Δ = 1.; F = 2.; γ = 1.;
    b = FockBasis(10);
    N = number(b);
    a = destroy(b);
    H = -Δ*N + F*(a + dagger(a));
    J = [sqrt(γ) * a];
    T = Float64[0.:0.1:10.;]

    # Initial conditions
    Ψ₀ = fockstate(b,0);

    # Test multithreaded_mcwf
    tout, Ψt = multithreaded_mcwf(T, Ψ₀, H, J, 100; seed=UInt(1), reltol=1e-7);
    tout2, Ψt2 = multithreaded_mcwf(T, Ψ₀, H, J, 100; seed=UInt(1), reltol=1e-7);
    @test Ψt == Ψt2

    tout, Ψt = multithreaded_mcwf(T, Ψ₀, H, J, 2; seed=UInt(1), reltol=1e-7)
    tout, Ψt2 = timeevolution.mcwf(T, Ψ₀, H, J; seed=UInt(1), reltol=1e-7)
    @test Ψt[1] == Ψt2
end

#=
Δ = 1.; F = 2.; γ = 1.;
b = FockBasis(10);
N = number(b);
a = destroy(b);
H = -Δ*N + F*(a + dagger(a));
J = [sqrt(γ) * a];
T = collect(0:10)

# Initial conditions
Ψ₀ = fockstate(b,0);
sol = pmcwf(T, Ψ₀, H, J;Ntrajectories=3, parallel_type=:threads, seed=UInt(1), reltol=1e-7)
sol = pmcwf(T, Ψ₀, H, J;Ntrajectories=2, parallel_type=:threads, reltol=1e-7, seed=UInt(1))
sol[1] == sol[2]
sol = pmcwf(T, Ψ₀, H, J;Ntrajectories=2, parallel_type=:none, progressbar=true, reltol=1e-7, seed=UInt(1))
params = Dict("pi" => π)
pmcwf(T, Ψ₀, H, J;Ntrajectories=3000, parallel_type=:pmap, progressbar=true, return_data=true,
    save_data=true, additional_data=params, fpath="E:/Documents/Julia scripts/Tests/Data/test14.jld2", reltol=1e-7, seed=UInt(1))
multithreaded_mcwf(T, Ψ₀, H, J, 2; seed=UInt(1), fout=(t,x)->expect(N,x)/norm(x)^2,reltol=1e-7)
file = jldopen("E:/Documents/Julia scripts/Tests/Data/test13.jld2","r")
file["trajs/2274"]
close(file)
tout, Ψt = timeevolution.mcwf(T, Ψ₀, H, J; seed=UInt(1), reltol=1e-7)
out_type = eltype(timeevolution.mcwf(T, Ψ₀, H, J; seed=UInt(1), reltol=1e-7)[2])
typeof(timeevolution.mcwf(T, Ψ₀, H, J; seed=UInt(1), reltol=1e-7))
sol = (Vector{Float64}(),Vector{out_type}())
multithreaded_mcwf(T, Ψ₀, H, J, 2; seed=UInt(1), reltol=1e-7)
using ProgressMeter
progress = Progress(10);
for i in 1:10
    ProgressMeter.next!(progress);
end


function read_saves(path::String, fname::String; trajrange=nothing, timerange=nothing)
    times = nothing
    sols = nothing;
    jldopen(path * fname * ".jld2", "r") do file
        fulltrajrange = 1:length(keys(file["trajs"]));
        if trajrange == nothing
            trajrange = fulltrajrange;
        else
            @assert trajrange[end] <= fulltrajrange[end] "The trajrange argument limits must not overpass those of the saved data"
        end
        fulltimerange = 1:length(file["t"]);
        if timerange == nothing
            timerange = fulltimerange;
        else
            @assert timerange[end] <= fulltimerange[end] "The timerange argument limits must not overpass those of the saved data"
        end
        times = file["t"][timerange]
        sols = [file["trajs/" * string(i)][t] for t in timerange, i in trajrange];
    end
    return times, sols
end

function split_last_integer(s::String)
    num::String = ""
    for c::Char in s
        try
            n = parse(Int, c);
            num *= c;
        catch
            num="";
        end
    end
    return SubString(s,1,length(s)-length(num)), num;
end;
function safe_save_name(path::String, fname::String)
    @assert ispath(path) "ERROR: accessing "*path*": No such file or directory"
    while isfile(path * fname * ".jld2")
        nfname::String, num::String = split_last_integer(fname);
        nfname *= num == "" ? "_1" : string(parse(Int, num)+1);
        fname = nfname;
    end
    return fname;
end;

"""
    tmap!(function, destination, collection)
Multithreaded version of [`map!`](@ref).

See also: [`map!`](@ref)

# Examples
```julia-repl
julia> src = collect(1:15);
julia> dst = similar(src);
julia> @btime map!(dst,src) do x
           Libc.systemsleep(1)
           x
       end
  15.039 s (0 allocations: 0 bytes)
julia> @btime tmap!(dst,src) do x
           Libc.systemsleep(1)
           x
       end
  3.006 s (1 allocation: 32 bytes)
```
"""
function tmap!(f, dst, src)
    @assert length(dst) == length(src) "Source and destination containers must have same lengths"
    Threads.@threads for i in eachindex(src)
        dst[i] = f(src[i]);
    end
end;
=#
