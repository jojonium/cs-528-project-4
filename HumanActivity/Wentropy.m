function Y = Wentropy(X)
    fs = 50;
    Y = nan(size(X, 1), 1);
    for i = 1:size(X, 1)
        Y(i) = pentropy(X(i, :), fs, 'Instantaneous', false);
    end
end