function [IMFS_EMD, optimal_K, u_opt] = part2_vmd_decomposition(...
    signal, alpha, tau, DC, init, tol, max_check_rounds)
load('entropy_low_point_result.mat', 'segment_times');

fprintf('=== Part 2: VMD Decomposition ===\n');

N = length(signal);

% Adaptive K selection (MDL stopping criterion)
K = 1;
mdl_values = [];
mse_values = [];
stop_flag = false;

while ~stop_flag
    [u, ~, ~] = VMD(signal, alpha, tau, K, DC, init, tol);

    % Signal reconstruction
    reconstructed = sum(u, 1)';
    mse = mean((signal - reconstructed).^2);
    mse_values(end+1) = mse;

    % Calculate MDL
    mdl = N * log(mse) + K * log(N);
    mdl_values(end+1) = mdl;

    if K >= 4
        % Current value is greater than minimum of previous three → possible rebound
        if mdl > min(mdl_values(end-3:end-1))
            % Continue for max_check_rounds additional rounds
            for extra = 1:max_check_rounds
                K = K + 1;
                fprintf('Additional VMD check, K = %d\n', K);
                [u, ~, ~] = VMD(signal, alpha, tau, K, DC, init, tol);

                reconstructed = sum(u, 1)';
                mse = mean((signal - reconstructed).^2);
                mse_values(end+1) = mse;

                mdl = N * log(mse) + K * log(N);
                mdl_values(end+1) = mdl;
            end
            stop_flag = true;
        end
    end
    K = K + 1;
end

% Select K corresponding to global minimum MDL
[~, idx_opt] = min(mdl_values);
optimal_K = idx_opt;
fprintf('Final adaptively selected K value: %d\n', optimal_K);

% Re-decompose using optimal K
[u_opt, ~, ~] = VMD(signal, alpha, tau, optimal_K, DC, init, tol);
IMFS_EMD = u_opt;
num_imfs = size(IMFS_EMD, 1);

% Save results
save('optimal_K.mat', 'optimal_K');
save('imfs_result.mat', 'IMFS_EMD');

end

%% ====== Subfunction Definitions ======

function entropy = calc_entropy(segment, num_bins)
    if nargin < 2
        num_bins = 30;
    end
    [hist_counts, bin_edges] = histcounts(segment, num_bins, 'Normalization', 'pdf');
    prob = hist_counts .* diff(bin_edges);
    prob = prob(prob > 0);
    entropy = -sum(prob .* log(prob));
end

function [positions, entropies, window_info] = sliding_entropy(signal, time, win_len, step, num_bins)
    if nargin < 5
        num_bins = 30;
    end
    entropies = [];
    positions = [];
    window_info = struct([]);
    for start = 1:step:(length(signal) - win_len + 1)
        seg = signal(start:start + win_len - 1);
        H = calc_entropy(seg, num_bins);
        entropies = [entropies; H];
        center_index = start + floor(win_len / 2);
        center_time = time(center_index);
        positions = [positions; center_time];
        win_info = struct(...
            'start_index', start, ...
            'end_index', start + win_len - 1, ...
            'start_time', time(start), ...
            'end_time', time(start + win_len - 1), ...
            'center_time', center_time, ...
            'entropy', H);
        window_info = [window_info; win_info];
    end
end