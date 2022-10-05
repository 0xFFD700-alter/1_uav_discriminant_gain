function [c_iter, a_iter] = solve_c(q_iter, dim, power, gain, sca)
% solve precoding strength c with SCA
    [c_iter, a_iter] = init_sca(q_iter, dim, power, gain);
    gain_iter = sum(a_iter);
    while 1
        cvx_begin
            cvx_solver mosek
            variable c(dim.K, dim.N) nonnegative
            variable a(1, dim.N) nonnegative
            maximize sum(a)
    
            subject to
                vstack = zeros(dim.K, dim.N);
                for k = 1: dim.K
                    vstack(k, :) = sum((q_iter - squeeze(power.w(k, :, :))) .^ 2, 2);
                end
                c .^ 2 .* (vstack + power.H ^ 2) .* power.E <= power.P * power.L_0;
                sum(c .^ 2 .* (vstack + power.H ^ 2) .* power.E, 2) <= power.P_bar * power.L_0 * dim.N;
        
                function_g = sum(c_iter) .^ 2 ./ a_iter;
                first_order_g = 2 * sum(c_iter) ./ a_iter .* sum(c - c_iter) ...
                                - (sum(c_iter) ./ a_iter) .^ 2 .* (a - a_iter);
                taylor_g = function_g + first_order_g;
                sum(c .^ 2 .* gain.delta) + sum(c) .^ 2 .* gain.sigma + gain.delta_0 <= taylor_g .* gain.u;
        cvx_end
            
        assert(strcmp(cvx_status, 'Solved'));
        c_iter = sca.momentum * c_iter + (1 - sca.momentum) * c;
        a_iter = sca.momentum * a_iter + (1 - sca.momentum) * a;

        if abs(cvx_optval - gain_iter) <= sca.epsilon
%             break
        end
        gain_iter = cvx_optval;
    end
end

