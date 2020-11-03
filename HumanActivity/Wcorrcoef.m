function output = Wcorrcoef(data, varNames)
    y = nan(size(data, 1), size(data, 3), size(data, 3));
    
    for i = 1:size(data, 1)
        y(i, :, :) = corrcoef(squeeze(data(i, :, :)));
    end
    
    splitmat = squeeze(num2cell(permute(y, [1 3 2]), [1 2]));
    
    for i = 1:size(data, 3)
       splitmat{i}(:, i) = [];
    end
    output = table(splitmat{:}, 'VariableNames', varNames);
end