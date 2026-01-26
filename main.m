% ====== Main Program: Signal Processing and IMF Selection ======
clc;
warning('off', 'all')

% ====== Parameter Settings ======
filepath = 'path';
data = load(filepath);
time = data(:,1);
signal = data(:,3);
signal_length = length(signal);
disp(signal_length);
fs = 100;
if signal_length > 500
    % --- Long Signal Parameters ---
    win_len = 100;              % Sliding window length
    step = 10;                  % Window step size
    min_peak_distance = 5;      % Minimum peak distance
    min_peak_prominence = 0.05; % Minimum peak prominence

    % VMD Parameters
    alpha = 2000;               % Bandwidth constraint
    tau = 0;                    % Noise tolerance
    DC = 0;                     % No DC part imposed
    init = 1;                   % Initialize frequencies uniformly
    tol = 1e-7;                 % Convergence tolerance
    max_check_rounds = 3;       % Maximum additional checking rounds after rebound

    % IMF Selection Parameters
    min_correlation = 0.1;      % Minimum correlation threshold
    thr = 0.4;                  % Correlation threshold

elseif  (200 < signal_length) && (signal_length < 500)
    % --- Short Signal Parameters ---
    win_len = 40;               % Shorter window
    step = 4;                   % Smaller step
    min_peak_distance = 6;      % Smaller peak distance
    min_peak_prominence = 0.01; % Lower prominence threshold

    % VMD Parameters
    alpha = 2000;               % More relaxed bandwidth parameter
    tau = 0;                    
    DC = 0;                     
    init = 1;                   
    tol = 1e-7;                 % Slightly relaxed tolerance
    max_check_rounds = 2;       % Fewer rebound checking rounds

    % IMF Selection Parameters
    min_correlation = 0.1;     % Lower correlation threshold
    thr = 0.3;
else 
    % --- Very Short Signal Parameters ---
    win_len = 20;               % Very short window
    step = 3;                   % Small step
    min_peak_distance = 4;      % Small peak distance
    min_peak_prominence = 0.001;% Very low prominence threshold

    % VMD Parameters
    alpha = 2000;               % More relaxed bandwidth parameter
    tau = 0;                    
    DC = 0;                     
    init = 1;                   
    tol = 1e-7;                 % Slightly relaxed tolerance
    max_check_rounds = 2;       % Fewer rebound checking rounds

    % IMF Selection Parameters
    min_correlation = 0.1;     % Lower correlation threshold
    thr = 0.2;
end

fprintf('=== Starting Signal Processing Pipeline ===\n');

%% ====== Part 1: Sliding Entropy Analysis ======
[time, signal, segment_times, entropy_times, ent, low_point_indices] = ...
    part1_sliding_entropy_analysis(filepath, win_len, step, min_peak_distance, min_peak_prominence);

%% ====== Part 2: VMD Decomposition ======
[IMFS_EMD, optimal_K, u_opt] = part2_vmd_decomposition(signal, alpha, tau, DC, init, tol, max_check_rounds);

%% ====== Part 3: Correlation Analysis and IMF Selection ======
[signal_imfs, noise_imfs, correlation_matrix] = part3_correlation_analysis(...
    time, signal, IMFS_EMD, segment_times, thr);

%% ====== Part 4: Signal Reconstruction ======
[reconstructed_signal, reconstruction_error] = part4_signal_reconstruction(...
    time, signal, IMFS_EMD, signal_imfs);

%% ====== Part 5: Similarity Evaluation ======
% similarity = part5_similarity_evaluation(filepath, reconstructed_signal, IMFS_EMD, noise_imfs);

%% ====== Part 6: Vehicle Detection ======
if contains(filepath, 'park')
    [accuracy, err_prob, miss_prob, detection_state, events] = part7_detection(time, reconstructed_signal, filepath, fs);
else
    [accuracy, err_prob, miss_prob, detection_state, events] = part6_detection(time, reconstructed_signal, filepath, fs);
end

fprintf('\n=== Processing Complete ===\n');