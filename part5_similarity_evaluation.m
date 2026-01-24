function similarity = part5_similarity_evaluation(filepath, reconstructed_signal, IMFS_EMD, noise_imfs)
    % ====== Part 5: Similarity Evaluation Between Reconstructed Signal and Clean Signal (Integrated Linear Calibration + Start Point Alignment) ======
    %
    % Input:
    %   filepath            - Original noisy signal file path
    %   reconstructed_signal- Reconstructed signal (Nx1)
    %   IMFS_EMD            - All IMF signal matrix (MxN)
    %   noise_imfs          - Noise IMF indices
    %
    % Output:
    %   similarity          - Structure containing metrics and signals for three versions

    %% ====== 1. Clean Signal Folder ======
    clean_folder = 'E:\Postdata\yan1\信号处理\运动信号分解\干净信号分解\split\label\';

    %% ====== 2. Generate Clean Signal File Path Based on Noisy Signal Filename ======
    [~, noisy_name, ext] = fileparts(filepath);
    tokens = regexp(noisy_name, '^(.*)\+.*(_[xyz])$', 'tokens');
    if isempty(tokens)
        error('Filename format does not match rules: %s', noisy_name);
    end
    base_name = tokens{1}{1};
    suffix    = tokens{1}{2};
    clean_name = [base_name, suffix, ext];
    clean_file = fullfile(clean_folder, clean_name);

    if ~isfile(clean_file)
        error('Clean signal file not found: %s', clean_file);
    end
    fprintf('\nCorresponding clean signal file: %s\n', clean_file);

    %% ====== 3. Read Clean Signal ======
    data_clean = load(clean_file);
    t_clean = data_clean(:,1);
    clean_signal = data_clean(:,3);

    % Force column vectors
    clean_signal = clean_signal(:);
    reconstructed_signal = reconstructed_signal(:);

    %% ====== 4. Align Signal Lengths ======
    N = min(length(reconstructed_signal), length(clean_signal));
    reconstructed_signal = reconstructed_signal(1:N);
    clean_signal = clean_signal(1:N);
    t_clean = t_clean(1:N);

    %% ====== 5. Original Metrics ======
    corr_pre = corrcoef(reconstructed_signal, clean_signal); corr_pre = corr_pre(1,2);
    mse_pre = mean((reconstructed_signal - clean_signal).^2);
    snr_pre = 10*log10(mean(clean_signal.^2) / mean((reconstructed_signal - clean_signal).^2));
    fprintf('Original reconstructed signal: Corr=%.4f, SNR=%.2f dB\n', corr_pre, snr_pre);

    %% ====== 6. Linear Regression Amplitude Calibration ======
    recon = reconstructed_signal - mean(reconstructed_signal);
    clean_zero = clean_signal - mean(clean_signal);
    X = [recon, ones(N,1)];
    params = X \ clean_zero;   % alpha, beta
    alpha = params(1); beta  = params(2);
    reconstructed_adj = alpha * recon + beta;

    corr_post = corrcoef(reconstructed_adj, clean_signal); corr_post = corr_post(1,2);
    mse_post = mean((reconstructed_adj - clean_signal).^2);
    snr_post = 10*log10(mean(clean_signal.^2) / mean((reconstructed_adj - clean_signal).^2));

    %% ====== 7. Start Point Alignment ======
    offset = reconstructed_signal(1) - clean_signal(1);
    reconstructed_aligned = reconstructed_signal - offset;

    corr_aligned = corrcoef(reconstructed_aligned, clean_signal); corr_aligned = corr_aligned(1,2);
    mse_aligned = mean((reconstructed_aligned - clean_signal).^2);
    snr_aligned = 10*log10(mean(clean_signal.^2) / mean((reconstructed_aligned - clean_signal).^2));
    fprintf('Start point aligned: Corr=%.4f, SNR=%.2f dB\n', corr_aligned, snr_aligned);

    %% ====== 8. Start + End Point Alignment ======
    start_offset = reconstructed_signal(1) - clean_signal(1);
    end_offset   = reconstructed_signal(end) - clean_signal(end);
    offset = (start_offset + end_offset)/2;
    reconstructed_aligned1 = reconstructed_signal - offset;
    
    corr_aligned = corrcoef(reconstructed_aligned1, clean_signal); corr_aligned = corr_aligned(1,2);
    mse_aligned2 = mean((reconstructed_aligned1 - clean_signal).^2);
    snr_aligned2 = 10*log10(mean(clean_signal.^2) / mean((reconstructed_aligned1 - clean_signal).^2));
    fprintf('Start+end point aligned: Corr=%.4f, SNR=%.2f dB\n', corr_aligned, snr_aligned2);

    % Calculate end offset
    end_offset = reconstructed_signal(end) - clean_signal(end);
    
    % End point alignment
    reconstructed_aligned2 = reconstructed_signal - end_offset;
    
    % Recalculate metrics
    corr_aligned = corrcoef(reconstructed_aligned2, clean_signal); 
    corr_aligned = corr_aligned(1,2);
    mse_aligned3 = mean((reconstructed_aligned2 - clean_signal).^2);
    snr_aligned3 = 10*log10(mean(clean_signal.^2) / mean((reconstructed_aligned2 - clean_signal).^2));
    
    fprintf('End point aligned: Corr=%.4f, SNR=%.2f dB\n', corr_aligned, snr_aligned3);

    %% ====== 9. Return Structure ======
    similarity = struct(...
        'Original', struct('Corr', corr_pre, 'MSE', mse_pre, 'SNR', snr_pre, 'Signal', reconstructed_signal), ...
        'LinearCalibrated', struct('Corr', corr_post, 'MSE', mse_post, 'SNR', snr_post, 'Signal', reconstructed_adj), ...
        'StartAligned', struct('Corr', corr_aligned, 'MSE', mse_aligned, 'SNR', snr_aligned, 'Signal', reconstructed_aligned), ...
        'StartEndAligned', struct('Corr', corr_aligned, 'MSE', mse_aligned2, 'SNR', snr_aligned2, 'Signal', reconstructed_aligned), ...
        'EndAligned', struct('Corr', corr_aligned, 'MSE', mse_aligned3, 'SNR', snr_aligned3, 'Signal', reconstructed_aligned) ...
    );
    
    %% ====== 10. Find Best SNR ======
    % Collect all SNRs
    snr_list = [similarity.Original.SNR, ...
                similarity.LinearCalibrated.SNR, ...
                similarity.StartAligned.SNR, ...
                similarity.StartEndAligned.SNR, ...
                similarity.EndAligned.SNR];

    % Find maximum and corresponding index
    [best_snr, best_idx] = max(snr_list);

    % Method name correspondence table
    methods = {'Original', 'LinearCalibrated', 'StartAligned', 'StartEndAligned', 'EndAligned'};
    best_method = methods{best_idx};

    % Save to structure
    similarity.Best.Method = best_method;
    similarity.Best.SNR    = best_snr;
    similarity.Best.Result = similarity.(best_method);

    fprintf('>>> Best method: %s, SNR = %.2f dB\n', best_method, best_snr);
end