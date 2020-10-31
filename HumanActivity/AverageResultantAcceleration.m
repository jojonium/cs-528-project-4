function ARA = AverageResultantAcceleration(X, Y, Z)
    ARA = mean(sqrt(X.^2 + Y.^2 + Z.^2), 2);
end