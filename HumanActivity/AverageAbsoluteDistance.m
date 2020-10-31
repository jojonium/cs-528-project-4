function Y = AverageAbsoluteDistance(X)
    Y = mean(abs(X - mean(X, 1)), 2);    
end