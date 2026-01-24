function [reconstructed_signal, reconstruction_error] = part4_signal_reconstruction(...
    time, signal, IMFS_EMD, signal_imfs)

fprintf('=== Part 4: Signal Reconstruction ===\n');

% Reconstruct signal (original, not denoised)
reconstructed_signal = sum(IMFS_EMD(signal_imfs, :), 1);


% Save final results
save('final_results.mat', 'signal_imfs', 'reconstructed_signal');

end