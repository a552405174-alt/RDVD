function [signal_imfs, noise_imfs, correlation_matrix] = part3_correlation_analysis(...
    time, signal, IMFS_EMD, segment_times, thr)

fprintf('=== Part 3: Correlation Analysis and IMF Selection ===\n');

num_imfs = size(IMFS_EMD, 1);
num_segments = length(segment_times) - 1;

% Calculate correlation matrix
correlation_matrix = zeros(num_imfs, num_segments);

for j = 1:num_segments
    % Start and end time of each segment
    start_time = segment_times(j);
    end_time = segment_times(j+1);
    
    % Find corresponding indices
    start_idx = find(time >= start_time, 1, 'first');
    end_idx = find(time <= end_time, 1, 'last');
    
    mixed_segment = signal(start_idx:end_idx);
    
    for i = 1:num_imfs
        imf_segment = IMFS_EMD(i, start_idx:end_idx);
        correlation_matrix(i, j) = corr(imf_segment', mixed_segment);
    end
end

% Save correlation matrix
save('correlation_matrix.mat', 'correlation_matrix');
fprintf('Correlation matrix calculation completed, dimensions: %d x %d\n', size(correlation_matrix, 1), size(correlation_matrix, 2));

S = correlation_matrix;
[M, K] = size(S);

imf1 = S(1, :);
imf2 = S(2, :);
signal_len = size(S,2);

use_imf3 = (M >= 10); 
if use_imf3
    imf3 = S(3, :);
end

if num_segments >= 5
    % Use findpeaks
    [pks1, locs1] = findpeaks(imf1, 'MinPeakDistance', 1);
    [pks2, locs2] = findpeaks(imf2, 'MinPeakDistance', 1);

    if use_imf3
        [pks3, locs3] = findpeaks(imf3, 'MinPeakDistance', 1);
    else
        locs3 = []; 
        pks3 = [];
    end

    locs1 = locs1(pks1 > thr & locs1 > 1 & locs1 < signal_len);
    locs2 = locs2(pks2 > thr & locs2 > 1 & locs2 < signal_len);
    if use_imf3
        locs3 = locs3(pks3 > thr & locs3 > 1 & locs3 < signal_len);
    end

    if numel(locs1) > 2
        [~, idx] = maxk(imf1(locs1), 2);
        locs1 = locs1(idx);
    end
    if numel(locs2) > 2
        [~, idx] = maxk(imf2(locs2), 2);
        locs2 = locs2(idx);
    end
    if use_imf3 && numel(locs3) > 2
        [~, idx] = maxk(imf3(locs3), 2);
        locs3 = locs3(idx);
    end

    all_idx = [locs1, locs2, locs3];
    unique_idx = unique(all_idx);
    counts = histc(all_idx, unique_idx);

    if numel(unique_idx) >= 2
        [~, sort_order] = maxk(counts, 2);
        peak_positions = sort(unique_idx(sort_order));
    elseif numel(unique_idx) == 1
        peak_positions = [unique_idx, unique_idx]; 
    else
        [~, tmp] = maxk(imf1, 2);
        peak_positions = sort(tmp);
    end

else
    [~, sorted_idx] = sort(imf1, 'descend');
    peak_positions = [];
    for i = 1:length(sorted_idx)
        if isempty(peak_positions)
            peak_positions = sorted_idx(i);
        elseif abs(sorted_idx(i) - peak_positions(1)) >= 2
            peak_positions(2) = sorted_idx(i);
            break;
        end
    end
    % If two points with sufficient spacing are not found, force take first two largest values
    if numel(peak_positions) < 2
        peak_positions = sorted_idx(1:2);
    end
    peak_positions = sort(peak_positions);

    % --- New feature: detect noise IMFs ---
    noise_imfs = [];
    for m = 4:M  
        for pos = peak_positions
            if S(m, pos) > imf1(pos)
                noise_imfs(end+1) = m;
                break; 
            end
        end
    end
end

if numel(peak_positions) == 1
    peak_positions = [peak_positions, peak_positions];
end

disp('Final selected peak positions:');
disp(peak_positions);


% --- Step 2: Traverse IMFs, check peak positions ---
signal_imfs = [];
noise_imfs = [];

for i = 1:M
    row = S(i, :);

    if i <= 2
        signal_imfs(end+1) = i;
    else
        % === IMF4 and beyond use new logic ===
        peak_vals = row(peak_positions);   % Values at peak positions
        max_peak_val = max(peak_vals);     % Maximum peak value
        row_max_val = max(row);            % Maximum value in entire IMF

        if row_max_val > max_peak_val
            noise_imfs(end+1) = i;
        elseif row_max_val < thr
            noise_imfs(end+1) = i;
        else
            signal_imfs(end+1) = i;
        end
    end
end

fprintf('Selected signal IMFs: %s\n', mat2str(signal_imfs'));
fprintf('Noise IMFs: %s\n', mat2str(noise_imfs'));

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

function [signal_imfs, noise_imfs, analysis_results] = advanced_imf_identification(S, min_correlation_threshold)
    % Advanced IMF identification algorithm
    [M, K] = size(S);
    
    % Calculate basic statistics
    mean_corr = mean(S, 2);
    std_corr = std(S, 0, 2);
    max_corr = max(S, [], 2);
    
    % Detect anomalously high correlations
    z_scores = (max_corr - mean(max_corr)) / std(max_corr);
    high_corr_outliers = find(z_scores > 0.5);
    
    % Calculate correlation stability scores
    stability_scores = 1 - (std_corr ./ (mean_corr + eps));
    
    % Calculate pattern consistency scores
    pattern_scores = zeros(M, 1);
    for i = 1:M
        if K > 1
            acf = autocorr(S(i, :), 1);
            pattern_scores(i) = abs(acf(2));
        else
            pattern_scores(i) = 0;
        end
    end
    
    % Calculate composite scores
    combined_scores = zeros(M, 1);
    for i = 1:M
        if ismember(i, high_corr_outliers)
            penalty = 0.5;
        else
            penalty = 1;
        end
        combined_scores(i) = penalty * (0.6 * mean_corr(i) + 0.2 * stability_scores(i) + 0.2 * pattern_scores(i));
    end
    
    % Adaptive threshold determination
    normalized_scores = (combined_scores - min(combined_scores)) / (max(combined_scores) - min(combined_scores));
    threshold = graythresh(normalized_scores);
    
    % Identify signal and noise IMFs
    signal_candidates = find(combined_scores >= threshold);
    
    % Ensure at least the first two IMFs are selected as signal
    if ~ismember(1, signal_candidates)
        signal_candidates = [1; signal_candidates];
    end
    if ~ismember(2, signal_candidates)
        signal_candidates = [2; signal_candidates];
    end
    
    signal_imfs = unique(signal_candidates);
    noise_imfs = setdiff(1:M, signal_imfs);
    
    % Store analysis results
    analysis_results.mean_corr = mean_corr;
    analysis_results.std_corr = std_corr;
    analysis_results.stability_scores = stability_scores;
    analysis_results.pattern_scores = pattern_scores;
    analysis_results.combined_scores = combined_scores;
    analysis_results.high_corr_outliers = high_corr_outliers;
    analysis_results.threshold = threshold;
    
    % Visualization
    visualize_advanced_analysis(S, analysis_results, signal_imfs, noise_imfs);
    
    fprintf('Identified signal IMFs: %s\n', mat2str(signal_imfs'));
    fprintf('Identified noise IMFs: %s\n', mat2str(noise_imfs'));
end

function visualize_advanced_analysis(S, results, signal_imfs, noise_imfs)
    [M, K] = size(S);
    
    figure('Position', [100, 100, 1400, 800]);
    
    % Correlation matrix heatmap
    subplot(2, 3, 1);
    imagesc(S);
    colorbar;
    xlabel('Segment Index');
    ylabel('IMF Index');
    title('Correlation Matrix Heatmap');
    
    hold on;
    for i = 1:length(signal_imfs)
        plot([0, K+1], [signal_imfs(i)-0.5, signal_imfs(i)-0.5], 'g-', 'LineWidth', 2);
    end
    for i = 1:length(noise_imfs)
        plot([0, K+1], [noise_imfs(i)-0.5, noise_imfs(i)-0.5], 'r-', 'LineWidth', 2);
    end
    
    % Mean correlation and standard deviation
    subplot(2, 3, 2);
    errorbar(1:M, results.mean_corr, results.std_corr, 'o', 'LineWidth', 1.5);
    hold on;
    
    for i = 1:length(signal_imfs)
        idx = signal_imfs(i);
        text(idx, results.mean_corr(idx), 'S', 'Color', 'green', 'FontWeight', 'bold', 'FontSize', 12);
    end
    for i = 1:length(noise_imfs)
        idx = noise_imfs(i);
        text(idx, results.mean_corr(idx), 'N', 'Color', 'red', 'FontWeight', 'bold', 'FontSize', 12);
    end
    
    xlabel('IMF Index');
    ylabel('Correlation');
    title('Mean Correlation ± Std Dev');
    grid on;
    
    % Composite scores
    subplot(2, 3, 3);
    bar(results.combined_scores);
    hold on;
    plot(xlim, [results.threshold results.threshold], 'r--', 'LineWidth', 2);
    
    for i = 1:length(signal_imfs)
        idx = signal_imfs(i);
        text(idx, results.combined_scores(idx), 'S', 'Color', 'green', 'FontWeight', 'bold', 'FontSize', 12);
    end
    for i = 1:length(noise_imfs)
        idx = noise_imfs(i);
        text(idx, results.combined_scores(idx), 'N', 'Color', 'red', 'FontWeight', 'bold', 'FontSize', 12);
    end
    
    xlabel('IMF Index');
    ylabel('Combined Score');
    title('Combined Scores with Threshold');
    grid on;
    
    % Stability scores
    subplot(2, 3, 4);
    bar(results.stability_scores);
    xlabel('IMF Index');
    ylabel('Stability Score');
    title('Stability Scores (1 - CV)');
    grid on;
    
    % Pattern scores
    subplot(2, 3, 5);
    bar(results.pattern_scores);
    xlabel('IMF Index');
    ylabel('Pattern Score');
    title('Pattern Scores (Autocorrelation)');
    grid on;
    
    % Correlation distribution comparison
    subplot(2, 3, 6);
    if ~isempty(signal_imfs)
        signal_correlations = reshape(S(signal_imfs, :), [], 1);
        histogram(signal_correlations, 20, 'FaceColor', 'green', 'FaceAlpha', 0.5);
        hold on;
    end
    if ~isempty(noise_imfs)
        noise_correlations = reshape(S(noise_imfs, :), [], 1);
        histogram(noise_correlations, 20, 'FaceColor', 'red', 'FaceAlpha', 0.5);
    end
    xlabel('Correlation Value');
    ylabel('Frequency');
    title('Correlation Distribution: Signal vs Noise IMFs');
    legend('Signal IMFs', 'Noise IMFs');
    grid on;
end