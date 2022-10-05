function [c_iter, a_iter] = init_sca(q_iter, dim, power, gain)
% SCA initialization
    vstack = zeros(dim.K, dim.N);
    for k = 1:dim.K
        vstack(k, :) = sum((q_iter - squeeze(power.w(k, :, :))) .^ 2, 2);
    end
    c_iter = sqrt(power.P .* power.L_0 ./ power.E ./ (vstack + power.H ^ 2) .* power.ratio) / 10;
    a_iter = sum(c_iter, 1) .^ 2 .* gain.u ./ (sum(c_iter, 1) .^ 2 .* gain.sigma + sum(gain.delta .* c_iter .^ 2, 1) + gain.delta_0) / 10;
end

