function [time, signal, segment_times, entropy_times, ent, low_point_indices] = ...
    part1_sliding_entropy_analysis(filepath, win_len, step, min_peak_distance, min_peak_prominence)

fprintf('=== Part 1: Sliding Entropy Analysis ===\n');

% Load data
data = load(filepath);
time = data(:, 1);
signal = data(:, 3);

% Calculate sliding entropy
[entropy_times, ent, window_info] = sliding_entropy(signal, time, win_len, step);

% Find low entropy points
[~, low_point_indices, ~, prominences] = findpeaks(-ent, ...
    'MinPeakDistance', min_peak_distance, ...
    'MinPeakProminence', min_peak_prominence);

if length(signal) > 500
    mean_entropy = 1 * mean(ent);
else
    mean_entropy = mean(ent);
end
valid_mask = ent(low_point_indices) <= mean_entropy;
if length(signal) > 500
    valid_mask = valid_mask & (entropy_times(low_point_indices) >= 90);
end
low_point_indices = low_point_indices(valid_mask);
low_point_times = entropy_times(low_point_indices);

fprintf('Number of detected low entropy points: %d\n', length(low_point_times));

% Organize segmentation time points
segment_times = [time(1); low_point_times; time(end)];

% Filter out segments that are too close
i = 1;
while i < length(segment_times) - 1
    if segment_times(i+1) - segment_times(i) < 5
        segment_times(i+1) = [];
    else
        i = i + 1;
    end
end

% Output segmentation information
num_segments = length(segment_times) - 1;
fprintf('Final number of segments: %d\n', num_segments);

% Save results
save('entropy_low_point_result.mat', 'num_segments', 'segment_times', 'ent', 'entropy_times');

% Plot results
% plot_entropy_results(time, signal, segment_times, entropy_times, ent, low_point_indices);

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

function plot_entropy_results(time, signal, segment_times, entropy_times, entropy_values, low_point_indices)
    
    figure;
    plot(time, signal, 'k-', 'LineWidth', 0.8);
    hold on;
    for i = 1:length(segment_times)
        xline(segment_times(i), 'r--', 'LineWidth', 1.2);
    end
    xlabel('Time (s)','FontName','Times New Roman','FontWeight','bold');
    ylabel('Amplitude','FontName','Times New Roman','FontWeight','bold');
    set(gca, 'FontName', 'Times New Roman','FontWeight','bold');

    figure;
    plot(entropy_times, -entropy_values, 'g-', 'LineWidth', 1.2);
    hold on;
    plot(entropy_times(low_point_indices), -entropy_values(low_point_indices), ...
        'ro', 'MarkerSize', 8, 'LineWidth', 2, 'MarkerFaceColor', 'r');
    xlabel('Time (s)','FontName','Times New Roman','FontWeight','bold');
    ylabel('Entropy Value','FontName','Times New Roman','FontWeight','bold');
    set(gca, 'FontName', 'Times New Roman','FontWeight','bold');
    legend('Entropy Curve', 'High Points', 'Location', 'northeast', 'FontName', 'Times New Roman');
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