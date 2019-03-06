using Distributed
# Get one process per vCPU (set less processes if RAM consumption is a concern)
num_processes = Sys.CPU_THREADS
if nprocs() < num_processes
    addprocs(num_processes - nprocs())
end

using ParallelMCWF, QuantumOptics, LinearAlgebra
# Operates the tensor product of all arguments
⨂(x) = length(x) > 1 ? reduce(⊗, x) : x[1];
# Operates the sum of all arguments
∑(x) = sum(x);
##

# SOME DEFINITIONS
# Simulation parameters
nb_trajs = 500;        # Number of trajectories
tspan = Array(range(0,stop=30,length=5));
n_site_max = 5;         # Local Hilbert space cut-off

# Model parameters
L = 3               # Length of the Bose-Hubbard chain
κ1 = 1.             # 1-body dissipation rate (To populate both parities)
κ2 = 1κ1            # 2-body dissipation rate
J  = 50κ1           # Tunneling rate
Δ  = -J             # Cavity detuning
U  = 100κ1          # Nonlinearity strength
F  = 0.             # 1-body driving strength
G  = 50κ1           # 2-photon driving strength

# Hamiltonian
lb = FockBasis(n_site_max); # Local Hilbert space basis
gb = lb⊗lb⊗lb;              # Global Hilbert space basis
Id = one(lb);               # Local Identity operator
la = destroy(lb);           # Local annihilation operator
lN = number(lb);            # Local number operator
# Local Hamiltonian (for a single site)
lH = -Δ*lN + U/2 *dagger(la)*lN*la + F*(la + dagger(la)) + G/2 *(la^2 + dagger(la)^2);
# Tight-binding interaction between the three sites
# This term is computed by summing all possible circular permutations of â†⊗â⊗𝟙̂⊗...⊗𝟙̂
# The Hermitian conjugate is latter added so as to have a Hermitian operator.
Hj = J/2 * ∑([⨂(circshift([dagger(la), la, fill(Id,L-2)...], i)) for i in 1:L]);
# Total Hamiltonian: sum of each site's local Hamiltonian plus the tunneling term and its Hermitian conjugate
# Local Hamiltonians are embedded into the global Hilbert space using the same technique
H = ∑([⨂(circshift([lH, fill(Id,L-1)...], i)) for i in 1:L]) + Hj + dagger(Hj);

# Some observables
lΠ = diagonaloperator(lb,Array(exp.(im*π .* diag(lN.data)))); # Local parity operator
# One could equally have used lΠ = exp(im*π*dense(lN)) but it has very poor performance
gN = ∑([⨂(circshift([lN, fill(Id,L-1)...], i)) for i in 1:L]); # Global number operator
gΠ = diagonaloperator(gb,Array(exp.(im*π .* diag(gN.data))));
##

# MONTE CARLO
# Liouvillians are represented as Arrays of jump operators
# 1-body dissipation jump operators
κ1𝒟 = sqrt(κ1) * [⨂(circshift([la, fill(Id,L-1)...], i)) for i in 1:L]
# 1-body dissipation jump operators
κ2𝒟 = sqrt(κ2) * [⨂(circshift([la^2, fill(Id,L-1)...], i)) for i in 1:L]

# Single ℤ₂-symmetric cavity
println("Single Z2-symmetric cavity:")
ψ0 = fockstate(lb,0) # Start from vacuum
G1  = 100κ1;
H1 = U/2 *dagger(la)*lN*la + F*(la + dagger(la)) + G1/2 *(la^2 + dagger(la)^2);
t, trajs = pmcwf(tspan, ψ0, H1, [sqrt(κ1)*la, sqrt(κ2)*la^2]; Ntrajectories=nb_trajs, parallel_type=:pmap)
# Steady-state kets
kets = [trajs[i][end] for i in 1:nb_trajs]
println("Parity = ",real.(kets_to_obs(lΠ, kets)))
ρ = kets_to_dm(kets)
println("Von Neumann entropy: S = ln($(exp(entropy_vn(ρ))))")

# L=3 global-Z2-symmetric chain
println("L=3 global-Z2-symmetric chain:")
ψ0 = fockstate(lb,0) ⊗ fockstate(lb,0) ⊗ fockstate(lb,0)
t, trajs = pmcwf(tspan, ψ0, H, [κ1𝒟; κ2𝒟]; Ntrajectories=nb_trajs, parallel_type=:pmap)
# Steady-state kets
kets = [trajs[i][end] for i in 1:nb_trajs]
println("Global parity = ",real.(kets_to_obs(gΠ, kets)))
ρ = kets_to_dm(kets)
println("Von Neumann entropy: S = ln($(exp(entropy_vn(ρ))))")

diag(ptrace(ρ, [2,3]).data)

using BenchmarkTools
@btime timeevolution.mcwf(tspan, ψ0, H, [κ1𝒟; κ2𝒟])
