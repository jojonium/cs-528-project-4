function Y = TimeBetweenPeaks(X)
    Y = nan(size(X, 1), 1);
    fs = 50;
    parfor trial = 1:size(X, 1)
        max_peak = max(X(trial, :));
        c = 0.9;       
        num_peaks = 0;
        
        while num_peaks < 3 && c >= 0.7
            thresh = c * max_peak;
            [~, locs] = findpeaks(X(trial, :), 'MinPeakHeight', thresh);
            num_peaks = numel(locs);
            c = c - 0.05;
        end
        if num_peaks >= 3
            mean_duration = mean(diff(locs) / fs * 1000);
        else
            mean_duration = 0;
        end
        Y(trial) = mean_duration;
    end
end