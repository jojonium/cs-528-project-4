function Y = BinnedDistribution(X)
    nbins = 10;
    Y = nan(size(X, 1), nbins);
    for trial = 1:size(X, 1)
        Y(trial, :) = histcounts(X(trial, :), nbins);
    end
end