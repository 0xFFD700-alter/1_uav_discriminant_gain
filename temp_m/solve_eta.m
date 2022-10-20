function eta_iter = solve_eta(q_iter, c_iter, dim, power)
    vstack = zeros(dim.K, dim.N);
    for k = 1:dim.K
        vstack(k, :) = sum((q_iter - squeeze(power.w(k, :, :))) .^ 2, 2);
    end
    a = min(min((vstack + power.H ^ 2) .* c_iter .^ 2 .* power.E ./ (power.P * power.L_0)));
    b = min(sum((vstack + power.H ^ 2) .* c_iter .^ 2 .* power.E, 2) ./ (power.P_bar * power.L_0 * dim.N));
    eta_iter = min(a, b);
end

