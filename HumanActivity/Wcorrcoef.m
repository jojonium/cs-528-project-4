function output = Wcorrcoef(data)
    y = nan(size(data, 1), size(data, 3), size(data, 3));
    
    for i = 1:size(data, 1)
        y(i, :, :) = corrcoef(squeeze(data(i, :, :)));
    end
    
    splitmat = num2cell(permute(y, [1 3 2]), [1 2]);
    output = table(splitmat{:});
end