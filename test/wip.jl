using SIGPROC, Plots, Dedisp, BenchmarkTools, CUDA, Clustering, DataFrames
using CUDA.CUFFT
pyplot()
default(; fmt=:png)

function cluster_candidates(dedisped, starting_times, dms;
                            snr_threshold=6,
                            radius=10, min_cluster_size=5)
    # Find peaks and bring back to CPU
    peak_idxs = Array(findall(x -> x > snr_threshold, dedisped))
    if length(peak_idxs) == 0
        return nothing
    end

    # Format the peaks matrix to hand over to dbscan
    peak_mat = Float32.(getindex.(peak_idxs, [1 2])')
    # Cluster!
    clusters = dbscan(peak_mat, radius; min_cluster_size=min_cluster_size)

    candidates = []

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
    return candidates
end

function find_transients(spectra, freqs, dms, starting_time, δt, n_gulp)
    # Plan is shared over dedisp interations
    f_min, f_max = extrema(freqs)
    n_chan = length(freqs)
    n_samp, _ = size(spectra)
    plan = plan_dedisp(cu(freqs), f_max, cu(dms), δt)

    # Preallocate output
    n_chan = fb.headers["nchans"]

    # Find optimum chunk size for this DM
    dm_max = maximum(dms)
    max_dm_delay = Δt(f_min, f_max, dm_max, δt)

    n_samp_out = n_gulp - max_dm_delay

    output = CUDA.zeros(n_samp_out, n_dm)

    candidates = DataFrame(; snr=Float32[], start=Float32[], dm=Float32[])
 
    for chunk_idx in 1:cld(n_samp, n_samp_out)
        # Upload chunk to GPU
        input_start_idx = n_samp_out * (chunk_idx-1) + 1
        input_stop_idx = input_start_idx + n_gulp - 1
        # If we go OOB, we need to like, pad with the mean
        input_chunk = spectra[input_start_idx:min(input_stop_idx,n_samp), :]
        if input_stop_idx > n_samp
            more_idxs = n_gulp - input_stop_idx + n_samp
            padding = fill(0f0,more_idxs,n_chan)
            input_chunk = vcat(input_chunk,padding)
        end
        # Upload to GPU
        pulse = cu(input_chunk)
        # Dedisperse
        out = standardize(dedisp!(output, pulse, plan))
        # Build offseted staring times
        starting_times = (input_start_idx:input_stop_idx) .* δt .+ starting_time
        # Find candidates in chunks
        local_cands = cluster_candidates(out, starting_times, dms)
        if !isnothing(local_cands)
            for cand in local_cands
                push!(candidates,cand)
            end
        end
    end

    return sort!(candidates,:snr,rev=true)
end

# Do the do
fb = Filterbank("/home/kiran/Downloads/frb_data/injectfrb_nfrb10_DM100-500_21sec_20220420-1005.fil")
freqs = collect(fb.data.dims[2])
dms = range(; start=10, stop=3000, length=1024)
find_transients(fb.data.data, freqs, dms, 0, fb.headers["tsamp"],100000)
