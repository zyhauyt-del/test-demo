% 约束非线性规划 — SQP算法求解 (MATLAB)
% 目标: min f(x) = x1² + x2² + x3² + 8
% s.t. 非线性不等式 + 等式约束 + 变量下界

clear; clc;

%% 问题定义
fun = @(x) x(1)^2 + x(2)^2 + x(3)^2 + 8;   % 目标函数
x0 = [1, 1, 1];                               % 初始点
lb = [0, 0, 0];                               % x >= 0

%% 求解器设置
options = optimoptions('fmincon', ...
    'Display', 'iter-detailed', ...            % 每次迭代详细输出
    'Algorithm', 'sqp', ...                    % 序列二次规划
    'OptimalityTolerance', 1e-6, ...
    'ConstraintTolerance', 1e-6, ...
    'StepTolerance', 1e-10, ...
    'MaxIterations', 500);

%% 调用 fmincon
[x_opt, fval, exitflag, output] = fmincon(fun, x0, [], [], [], [], lb, [], @nonlin_con, options);

%% 结果输出
fprintf('\n========== 优化结果 ==========\n');
fprintf('最优解:   x* = [%.6f, %.6f, %.6f]\n', x_opt);
fprintf('最优值:   f* = %.6f\n', fval);
fprintf('迭代次数: %d\n', output.iterations);
fprintf('函数评估: %d\n', output.funcCount);
fprintf('退出标志: %d (1=收敛)\n', exitflag);

% 约束验证
[c, ceq] = nonlin_con(x_opt);
fprintf('\n===== 约束验证 =====\n');
fprintf('不等式约束 c ≤ 0:\n');
for i = 1:length(c)
    fprintf('  c(%d) = %.6f (需 ≤ 0) %s\n', i, c(i), ...
        iif(c(i) <= 1e-6, '✓', '✗'));
end
fprintf('等式约束 ceq = 0:\n');
for i = 1:length(ceq)
    fprintf('  ceq(%d) = %.6f (需 = 0) %s\n', i, ceq(i), ...
        iif(abs(ceq(i)) <= 1e-6, '✓', '✗'));
end

%% 非线性约束函数
function [c, ceq] = nonlin_con(x)
    % 不等式约束: c(x) ≤ 0
    c(1) = -(x(1)^2 - x(2) + x(3)^2);        % 即 x1² - x2 + x3² ≥ 0
    c(2) = x(1) + x(2)^2 + x(3)^3 - 20;       % 即 x1 + x2² + x3³ ≤ 20

    % 等式约束: ceq(x) = 0
    ceq(1) = -x(1) - x(2)^2 + 2;               % 即 x1 + x2² = 2
    ceq(2) = x(2) + 2*x(3)^2 - 3;              % 即 x2 + 2x3² = 3
end

%% 辅助函数: 三元条件
function s = iif(cond, t, f)
    if cond
        s = t;
    else
        s = f;
    end
end
