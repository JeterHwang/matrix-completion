function [A, b, xsol, pv, pq, ref, line] = makePSSEdata( data )
%MAKESECON Given MATPOWER case file, make A, b, C matrices so that the
%standard state estimation problem can be posed as the semidefinite program
%   min   c' * x
%   s.t.  A' * x  = b
%         mat(x) >= 0 
% MATPOWER must be in the search path.

% See also quadconstr

% Input
%    data  -- MATPOWER case file
% Output
%  A,b,c,K -- Semidefinite program in SeDuMi format
%             The first 2*nbus constraints in A and b correspond to the
%             usual power flow problem. The remaining constraints are the
%             remaining state estimation measurements
%    xsol  -- Unique rank-1 solution
%             for the powerflow problem

[PQ, PV, REF, NONE, BUS_I, BUS_TYPE, PD, QD, GS, BS, BUS_AREA, VM, ...
    VA, BASE_KV, ZONE, VMAX, VMIN, LAM_P, LAM_Q, MU_VMAX, MU_VMIN] = idx_bus;
[F_BUS, T_BUS, BR_R, BR_X, BR_B, RATE_A, RATE_B, RATE_C, ...
    TAP, SHIFT, BR_STATUS, PF, QF, PT, QT, MU_SF, MU_ST, ...
    ANGMIN, ANGMAX, MU_ANGMIN, MU_ANGMAX] = idx_brch;
[GEN_BUS, PG, QG, QMAX, QMIN, VG, MBASE, GEN_STATUS, PMAX, PMIN, ...
    MU_PMAX, MU_PMIN, MU_QMAX, MU_QMIN, PC1, PC2, QC1MIN, QC1MAX, ...
    QC2MIN, QC2MAX, RAMP_AGC, RAMP_10, RAMP_30, RAMP_Q, APF] = idx_gen;

% Presolve powerflow.
data = runpf(data);

%==========================================================================
% Standard matpower commands
mpc = loadcase(data);
mpc = ext2int(mpc);
[baseMVA, bus, gen, branch] = deal(mpc.baseMVA, mpc.bus, mpc.gen, mpc.branch);
[ref, pv, pq] = bustypes(bus, gen);

% make sure that slack bus angle is zero
bus(:, VA) = bus(:, VA) - bus(ref,VA);

V0  = bus(:, VM) .* exp(sqrt(-1) * pi/180 * bus(:, VA));
%==========================================================================

% Bus info
nbus = numel(V0);
[Ybus, Yf, Yt] = makeYbus(baseMVA, bus, branch);
Sbus = V0 .* conj(Ybus*V0);

% Line info
f = branch(:, F_BUS);
t = branch(:, T_BUS);
Sf = V0(f) .* conj(Yf*V0);
St = V0(t) .* conj(Yt*V0);

% Ap  - Nodal real power
% Aq  - Nodal reactive power
% Av  - Nodal voltage
% Apf - Line real power (from)
% Aqf - Line reactive power (from)
% Apt - Line real power (to)
% Aqt - Line reactive power (to)

[Ap, Aq, Av, Apf, Aqf, Apt, Aqt] = quadconstr(Ybus, Yf, f, Yt, t);

% Output
A = [Ap, Aq, Av, Apf, Aqf, Apt, Aqt];
b = [real(Sbus); imag(Sbus);
     abs(V0).^2; 0;
     real(Sf); imag(Sf);
     real(St); imag(St);
     ];
v = [real(V0); imag(V0)];

% Delete the slack bus imaginary component (which is set to zero)
idx1 = nbus + ref;
idx2 = false(2*nbus); 
idx2(:,idx1) = true; idx2(idx1,:) = true;
v(idx1) = [];
A(idx2(:),:) = [];

% Generate cost function to guarantee rank-1 solution
C = eye(2*nbus-1) - (v*v') / (v'*v);
c = C(:);

% SeDuMi Cone
K = struct('l',0,'s',nbus*2-1);
xsol = v;

% The original powerflow problem
ref = ismember(1:nbus, ref);
pv = ismember(1:nbus, pv);
pq = ismember(1:nbus, pq);
sel = find([pv | pq, pq, pv | ref]);
assert(numel(sel) == (2*nbus -1), 'Power flow problem must always have 2*nbus-1 constraints');

% Reshuffle the A matrix to put the powerflow constraints as the first m
% constraints
sel = ismember(1:size(A,2), sel);
A = [A(:, sel), A(:, ~sel)];
b = [b(sel); b(~sel)];

% Other outputs

[ref, pv, pq] = bustypes(bus, gen);
line = [branch(:,F_BUS), branch(:,T_BUS)];

end




