using SIGPROC, Plots, Dedisp, BenchmarkTools, CUDA, Clustering, DataFrames
using CUDA.CUFFT
pyplot()
default(; fmt=:png)

# Read Filterbank Data
fb = Filterbank("/home/kiran/Downloads/frb_data/injectfrb_nfrb10_DM100-500_21sec_20220420-1005.fil")

# Build plan
freqs = cu(collect(fb.data.dims[2]))
f_min, f_max = extrema(fb.data.dims[2])
n_samp = length(fb.data.dims[1])
#n_samp = 100000
δt = fb.headers["tsamp"]
t_total = n_samp * δt

dm_max = 2000
dm_min = 10
n_dm = 1024

dms = cu(collect(range(; start=dm_min, stop=dm_max, length=n_dm)))
plan = plan_dedisp(freqs, f_max, dms, δt)

# Preallocate output
n_chan = fb.headers["nchans"]
n_samp_out = n_samp ÷ 2
output = CUDA.zeros(n_samp_out, n_dm)

# Dedisperse
pulse = cu(fb.data.data[1:n_samp, :])
out = standardize(dedisp!(output, pulse, plan))

# Pretty Plot
#heatmap((δt .* (1:(n_samp ÷ 2))), Array(dms), Array(standardize(out))'; clims=(0, 45),
#        xlabel="Starting Time Offset (s)", ylabel="DM", c=:jet)
# heatmap(Array(out))

function cluster_candidates(dedisped, starting_times, dms;
                            snr_threshold=6,
                            radius=10, min_cluster_size=5)
    # Find peaks and bring back to CPU
    peak_idxs = Array(findall(x -> x > snr_threshold, dedisped))
    # Format the peaks matrix to hand over to dbscan
    peak_mat = Float32.(getindex.(peak_idxs, [1 2])')
    # Cluster!
    clusters = dbscan(peak_mat, radius; min_cluster_size=min_cluster_size)
    # Initialize candidates data frame
    candidates = DataFrame(; snr=Float32[], start=Float32[], dm=Float32[])
    # Fill DF
    for cluster in clusters
        # This is which cols from peak_mat are the indices in the cluster
        cluster_idxs = cluster.core_indices

        snrs = dedisped[peak_idxs[cluster_idxs]]

        max_snr, local_idx = findmax(snrs)
        max_snr_idx = peak_idxs[cluster_idxs][local_idx]

        starting_time = starting_times[Tuple(max_snr_idx)[1]]
        dm = dms[Tuple(max_snr_idx)[2]]

        push!(candidates, [max_snr, starting_time, dm])
    end
    # Sort by SNR
    return sort(candidates, :snr; rev=true)
end

cluster_candidates(out, (1:n_samp_out) .* δt, Array(dms))
