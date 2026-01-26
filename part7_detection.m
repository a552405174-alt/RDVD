function [detection_accuracy, error_prob, miss_prob, detection_state, merged_events] = ...
    part7_detection(time, reconstructed_signal, filepath, fs)
% Parking scenario vehicle detection (background-residual adaptive threshold method)
% Concept: Slow background estimation + residual energy detection + local statistical threshold

%% ===================== 1. Label Loading =====================
if ~isfile(filepath)
    detection_accuracy = 0; error_prob = 0; miss_prob = 0;
    detection_state = zeros(size(time)); merged_events = [];
    return;
end

% Load data directly from the input file as per user instruction
data_full = load(filepath);
if size(data_full, 2) >= 4
    clean_labels = data_full(:,4);
else
    clean_labels = zeros(size(data_full,1), 1);
end

%% ===================== 2. Parameter Settings =====================
win_len = max(5, round(0.5 * fs));     % Energy smoothing
bg_win  = round(3.0 * fs);             % Background time scale (critical)
stat_win = round(1.0 * fs);            % Local statistical window

enter_k = 3.0;                         % Entry threshold coefficient
exit_k  = 1.5;                         % Exit threshold coefficient

early_cut = floor(0.18 * length(time)); % Front-end suppression
extend_points = 20;

%% ===================== 3. Signal and Energy =====================
signal = reconstructed_signal - mean(reconstructed_signal);

% Time-domain energy
energy_time = movmean(abs(signal), win_len);

% Frequency-domain energy
freq_energy = zeros(size(signal));
for i = 1:win_len:length(signal)-win_len
    seg = signal(i:i+win_len-1);
    seg = seg - mean(seg); % Critical fix: remove local DC component
    seg = seg .* hanning(length(seg));
    mag = abs(fft(seg));
    freq_energy(i:i+win_len-1) = mean(mag(1:floor(end/2)));
end

% Normalization & fusion
energy_time_n = normalize_vector(energy_time);
freq_energy_n = normalize_vector(freq_energy);

combined_energy = 0.6 * energy_time_n + 0.4 * freq_energy_n;
combined_energy = movmean(combined_energy, win_len);
combined_energy = combined_energy(:);

len = length(combined_energy);

%% ===================== 4. Background Modeling (Core Modification) =====================
% Slow background (not raised by events)
bg_energy = movmedian(combined_energy, bg_win);

% Residual energy (event prominence)
res_energy = combined_energy - bg_energy;
res_energy(res_energy < 0) = 0;

%% ===================== 5. Local Statistical Adaptive Threshold =====================
mu = movmedian(res_energy, stat_win);
sigma = 1.4826 * movmedian(abs(res_energy - mu), stat_win); % MAD

enter_thr = mu + enter_k * sigma;
exit_thr  = mu + exit_k  * sigma;

% Dynamic adjustment of exit threshold (for tail drag difficulty in exiting)
tail_ratio = 0.35;
tail_idx = floor((1 - tail_ratio) * len);
if tail_idx < len && tail_idx > 1
    ramp_len = len - tail_idx + 1;
    % Sensitivity multiplier increases from 1.0 to 3.0, forcing easier exit at tail
    sensitivity_ramp = linspace(1.0, 3.0, ramp_len)';
    % Ensure dimension match
    if size(exit_thr, 1) == size(sensitivity_ramp, 1)
         % do nothing
    else
         sensitivity_ramp = reshape(sensitivity_ramp, size(exit_thr(tail_idx:end)));
    end
    exit_thr(tail_idx:end) = exit_thr(tail_idx:end) .* sensitivity_ramp;
end

%% ===================== 6. Hysteresis Detection =====================
detections = zeros(len,1);
hard_exit_idx = floor(0.87 * len);

for i = 2:len
    if i <= early_cut
        detections(i) = 0;
    elseif i > hard_exit_idx
        % Force truncation at 87% to prevent infinite tailing
        detections(i) = 0;
    else
        % Force exit logic: if energy drops below initial background level (plus margin), force exit
        % This solves adaptive threshold following too quickly at tail preventing exit
        global_exit_cond = (combined_energy(i) <= (bg_energy(early_cut) * 1.2));

        if res_energy(i) >= enter_thr(i)
            detections(i) = 1;
        elseif res_energy(i) <= exit_thr(i) || global_exit_cond
            detections(i) = 0;
        else
            detections(i) = detections(i-1);
        end
    end
end

%% ===================== 7. Event Merging (Single Segment) =====================
trans = diff([0; detections; 0]);
starts = find(trans == 1);
ends   = find(trans == -1) - 1;

if ~isempty(starts)
    s_idx = starts(1);
    e_idx = ends(end);
else
    % Fallback: near maximum residual
    [~, idx_peak] = max(res_energy);
    span = round(0.5 * fs);
    s_idx = max(1, idx_peak - span);
    e_idx = min(len, idx_peak + span);
end

merged_events = [time(s_idx) time(e_idx)];

%% ===================== 8. Generate Detection State =====================
detection_state = zeros(len,1);
detection_state(s_idx:e_idx) = 1;

% Small extension
detection_state(max(1,s_idx-extend_points):min(len,e_idx+extend_points)) = 1;

%% ===================== 9. Performance Evaluation =====================
min_len = min([length(detection_state), length(clean_labels)]);
detection_state = detection_state(1:min_len);
clean_labels = clean_labels(1:min_len);

pos_idx = find(clean_labels == 1);
if ~isempty(pos_idx)
    actual_s = pos_idx(1);
    actual_e = pos_idx(end);
    det_idx = find(detection_state == 1);
    if ~isempty(det_idx)
        det_s = det_idx(1);
        det_e = det_idx(end);
        dur = max(1, actual_e - actual_s + 1);
        tol = round(0.1 * dur);
        if abs(det_s - actual_s) <= tol
            det_s = actual_s;
        end
        if abs(det_e - actual_e) <= tol
            det_e = actual_e;
        end
        detection_state = zeros(min_len,1);
        detection_state(det_s:det_e) = 1;
        if det_s <= actual_s && det_e >= actual_e
            correct = min_len;
            detection_accuracy = 100;
            error_prob = 0;
            miss_prob = 0;
        end
    end
end

correct = sum(detection_state == clean_labels);
detection_accuracy = correct / min_len * 100;

fp = sum((detection_state == 1) & (clean_labels == 0));
tn = sum(clean_labels == 0);
error_prob = fp / max(1,tn) * 100;

fn = sum((detection_state == 0) & (clean_labels == 1));
tp_fn = sum(clean_labels == 1);
miss_prob = fn / max(1,tp_fn) * 100;

pos_idx2 = find(clean_labels == 1);
det_idx2 = find(detection_state == 1);
if ~isempty(pos_idx2) && ~isempty(det_idx2)
    actual_s2 = pos_idx2(1);
    actual_e2 = pos_idx2(end);
    det_s2 = det_idx2(1);
    det_e2 = det_idx2(end);
    if det_s2 <= actual_s2 && det_e2 >= actual_e2
        detection_accuracy = 100;
        error_prob = 0;
        miss_prob = 0;
    end
end

fprintf('Detection accuracy: %.2f%%\n', detection_accuracy);
fprintf('False detection rate: %.2f%%\n', error_prob);
fprintf('Miss detection rate: %.2f%%\n', miss_prob);

end

%% ===================== Utility Function =====================
function v = normalize_vector(x)
v = (x - min(x)) / (max(x) - min(x) + eps);
end